

"""
RAG (Retrieval-Augmented Generation) service.

Orchestrates the full retrieval pipeline:
    1. Embed a plain-text query using Amazon Nova Embed (via Bedrock)
    2. Query ChromaDB for the top-k most similar document chunks
    3. Optionally re-rank results by a relevance threshold
    4. Return structured, agent-ready context objects

This service sits between the vector store (infrastructure) and the
retriever tool (core/tools), keeping both layers clean and focused.
"""
import json
import uuid
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from src.infrastructure.vector_store.base import BaseVectorStore
from utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Amazon Nova Embed model ID on Bedrock
# Produces 1024-dimensional dense vectors
NOVA_EMBED_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"

# Default retrieval settings
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.35  # Chunks below this score are filtered out


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

class RetrievedChunk:
    """
    A single retrieved document chunk returned by the RAG service.
    Structured so the agent always receives a consistent, readable object.
    """

    def __init__(
        self,
        chunk_id: str,
        content: str,
        score: float,
        source: str,
        doc_type: str,
        metadata: Dict[str, Any],
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.source = source
        self.doc_type = doc_type
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSON responses and tool output."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "doc_type": self.doc_type,
            "metadata": self.metadata,
        }

    def to_context_string(self) -> str:
        """
        Format the chunk as a readable context block for injection
        into an LLM prompt. Each chunk is clearly delimited so the
        model can cite sources accurately.
        """
        return (
            f"[SOURCE: {self.source} | TYPE: {self.doc_type} | RELEVANCE: {self.score}]\n"
            f"{self.content}\n"
        )


# ---------------------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------------------

class RAGService:
    """
    Retrieval-Augmented Generation service.

    Wraps embedding generation and vector store querying into a single
    high-level interface consumed by the retriever tool.

    Args:
        vector_store: An initialized BaseVectorStore instance (ChromaDB)
        aws_region: AWS region where the Bedrock endpoint is hosted
        embed_model_id: Bedrock embedding model ID (defaults to Nova Embed)
        default_top_k: Default number of chunks to retrieve per query
        score_threshold: Minimum similarity score to include a chunk
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        aws_region: str = "us-east-1",
        embed_model_id: str = NOVA_EMBED_MODEL_ID,
        default_top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ):
        self.vector_store = vector_store
        self.embed_model_id = embed_model_id
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold

        # Bedrock runtime client — used only for embedding, not generation
        self._bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region,
        )

        logger.info(
            f"RAGService initialized | "
            f"embed_model={embed_model_id} | "
            f"top_k={default_top_k} | "
            f"threshold={score_threshold}"
        )

    # ------------------------------------------------------------------
    # Public API — consumed by the retriever tool
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Full retrieval pipeline: embed query → search → filter → return chunks.

        Args:
            query: Plain-text clinical query from the agent
            top_k: Number of chunks to retrieve (overrides instance default)
            filters: Optional metadata filters passed to ChromaDB
                     e.g. {"doc_type": "clinical_guideline"}
                          {"specialty": "cardiology"}
                          {"$and": [{"doc_type": "protocol"}, {"specialty": "icu"}]}
            score_threshold: Override minimum similarity score for this call

        Returns:
            List of RetrievedChunk objects, sorted by relevance (highest first).
            Returns an empty list if retrieval fails — never raises to the agent.
        """
        if not query or not query.strip():
            logger.warning("RAGService.retrieve called with empty query")
            return []

        k = top_k or self.default_top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        try:
            # Step 1: Embed the query
            query_embedding = await self._embed_text(query)
            if not query_embedding:
                logger.error("Embedding returned empty vector — aborting retrieval")
                return []

            # Step 2: Query the vector store
            raw_results = await self.vector_store.query(
                query_embedding=query_embedding,
                top_k=k,
                filters=filters,
            )

            if not raw_results:
                logger.info(f"No results found for query: '{query[:80]}...'")
                return []

            # Step 3: Filter by score threshold and build RetrievedChunk objects
            chunks = self._build_chunks(raw_results, threshold)

            logger.info(
                f"Retrieved {len(chunks)}/{len(raw_results)} chunks above threshold "
                f"({threshold}) for query: '{query[:60]}...'"
            )

            return chunks

        except Exception as e:
            logger.error(f"RAGService.retrieve failed: {e}", exc_info=True)
            return []

    async def retrieve_as_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> str:
        """
        Retrieve chunks and format them as a single context string
        ready to be injected directly into an LLM prompt.

        Args:
            query: Plain-text clinical query
            top_k: Number of chunks to retrieve
            filters: Optional metadata filters
            score_threshold: Minimum similarity score

        Returns:
            Formatted multi-chunk context string, or empty string if no results
        """
        chunks = await self.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )

        if not chunks:
            return ""

        header = f"Retrieved {len(chunks)} relevant document(s):\n\n"
        body = "\n---\n".join(chunk.to_context_string() for chunk in chunks)
        return header + body

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def _embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string using Amazon Nova Embed via Bedrock.

        Nova Embed expects:
            {"inputText": "..."}

        And returns:
            {"embedding": [...], "inputTextTokenCount": N}

        Args:
            text: Text to embed

        Returns:
            1024-dimensional embedding vector, or empty list on failure
        """
        try:
            body = json.dumps({"inputText": text})

            response = self._bedrock.invoke_model(
                modelId=self.embed_model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            embedding = response_body.get("embedding", [])

            if not embedding:
                logger.error("Nova Embed returned an empty embedding vector")
                return []

            logger.debug(f"Embedded text ({len(embedding)}-dim): '{text[:60]}...'")
            return embedding

        except ClientError as e:
            logger.error(f"Bedrock embedding ClientError: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Embedding failed: {e}", exc_info=True)
            return []

    async def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Embed a batch of texts. Used by the ingestion pipeline (utils/ingest.py).
        Calls Nova Embed individually per text — Bedrock does not support
        batch embedding in a single call for Nova Embed.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors in the same order as input texts.
            Failed embeddings are returned as empty lists — caller should
            validate before upserting.
        """
        embeddings = []
        for i, text in enumerate(texts):
            embedding = await self._embed_text(text)
            embeddings.append(embedding)

            if (i + 1) % 10 == 0:
                logger.info(f"Embedded {i + 1}/{len(texts)} documents...")

        logger.info(f"Batch embedding complete: {len(embeddings)} vectors generated")
        return embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chunks(
        self,
        raw_results: List[Dict[str, Any]],
        threshold: float,
    ) -> List[RetrievedChunk]:
        """
        Convert raw vector store results into RetrievedChunk objects,
        filtering out anything below the score threshold.

        Pulls `source` and `doc_type` from metadata as first-class fields
        since they are the most useful for the agent to reason over.
        Both fields fall back to "unknown" if not present in metadata.

        Args:
            raw_results: Raw dicts from ChromaVectorStore.query()
            threshold: Minimum score to include

        Returns:
            Filtered, structured list of RetrievedChunk objects
        """
        chunks = []

        for result in raw_results:
            score = result.get("score", 0.0)

            if score < threshold:
                logger.debug(
                    f"Chunk '{result.get('id')}' filtered out — "
                    f"score {score} below threshold {threshold}"
                )
                continue

            metadata = result.get("metadata", {})

            chunks.append(
                RetrievedChunk(
                    chunk_id=result.get("id", str(uuid.uuid4())),
                    content=result.get("content", ""),
                    score=score,
                    source=metadata.get("source", "unknown"),
                    doc_type=metadata.get("doc_type", "unknown"),
                    metadata=metadata,
                )
            )

        return chunks