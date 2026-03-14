"""
RAG (Retrieval-Augmented Generation) service.

Orchestrates the full retrieval and ingestion pipeline:

Retrieval:
    1. Embed a plain-text query via the injected embedding model
    2. Query ChromaDB for the top-k most similar document chunks
    3. Filter results by a relevance score threshold
    4. Return structured, agent-ready RetrievedChunk objects

Ingestion (moved from utils/ingest.py — called by the upload endpoint):
    1. Accept pre-extracted text and metadata (PDF parsing stays in utils)
    2. Chunk the text with a sentence-aware sliding window
    3. Embed all chunks via the injected embedding model
    4. Upsert into the vector store

The service depends on abstractions only:
    - BaseEmbeddingModel  (not BedrockEmbeddingModel directly)
    - BaseVectorStore     (not ChromaVectorStore directly)

This keeps the service fully testable and provider-agnostic.
"""
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.infrastructure.embedding_models.base import BaseEmbeddingModel
from src.infrastructure.vector_store.base import BaseVectorStore
from utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.35


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
        Format as a readable context block for LLM prompt injection.
        Each chunk is clearly delimited so the model can cite sources.
        """
        return (
            f"[SOURCE: {self.source} | TYPE: {self.doc_type} | "
            f"RELEVANCE: {self.score:.3f}]\n"
            f"{self.content}\n"
        )


# ---------------------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------------------

class RAGService:
    """
    Retrieval-Augmented Generation service.

    Depends on injected abstractions only — no direct boto3 / ChromaDB imports.

    Args:
        embedding_model: Any BaseEmbeddingModel implementation
        vector_store: Any BaseVectorStore implementation
        default_top_k: Default number of chunks to retrieve per query
        score_threshold: Minimum similarity score to include a chunk
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        default_top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold

        logger.info(
            f"RAGService initialized | "
            f"embedding_model={embedding_model.__class__.__name__} | "
            f"top_k={default_top_k} | threshold={score_threshold}"
        )

    # ------------------------------------------------------------------
    # Retrieval — consumed by tools and the RAG router
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
            query: Plain-text clinical query from the agent or endpoint
            top_k: Number of chunks to retrieve (overrides instance default)
            filters: Optional metadata filters passed to the vector store
                     e.g. {"doc_type": "clinical_guideline"}
                          {"specialty": "cardiology"}
                          {"$and": [{"doc_type": "protocol"}, {"specialty": "icu"}]}
            score_threshold: Override minimum similarity score for this call

        Returns:
            List of RetrievedChunk objects sorted by relevance (highest first).
            Returns an empty list if retrieval fails — never raises to callers.
        """
        if not query or not query.strip():
            logger.warning("RAGService.retrieve called with empty query")
            return []

        k = top_k or self.default_top_k
        threshold = (
            score_threshold if score_threshold is not None else self.score_threshold
        )

        try:
            query_embedding = await self.embedding_model.embed_text(query)
            if not query_embedding:
                logger.error("Embedding returned empty vector — aborting retrieval")
                return []

            raw_results = await self.vector_store.query(
                query_embedding=query_embedding,
                top_k=k,
                filters=filters,
            )

            if not raw_results:
                logger.info(f"No results found for query: '{query[:80]}'")
                return []

            chunks = self._build_chunks(raw_results, threshold)

            logger.info(
                f"Retrieved {len(chunks)}/{len(raw_results)} chunks above "
                f"threshold ({threshold}) for query: '{query[:60]}'"
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
        Retrieve chunks and return them as a single formatted context string
        ready to be injected directly into an LLM prompt.

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
    # Ingestion — called by the upload endpoint via the router
    # ------------------------------------------------------------------

    async def ingest_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id_prefix: str,
    ) -> int:
        """
        Embed and upsert pre-processed document chunks into the vector store.

        PDF parsing and chunking live in utils/ingest.py. This method
        receives already-chunked text so the service stays focused on
        the embedding + storage concern.

        Args:
            text: Full document text (will be chunked internally)
            metadata: Base metadata dict for the document
                      (source filename, doc_type, and any extracted fields)
            doc_id_prefix: Stable prefix for chunk IDs, e.g. "ng_001_patient_record"

        Returns:
            Number of chunks successfully upserted, or 0 on failure
        """
        # Lazy import to avoid circular dependency — ingest helpers are pure utils
        from utils.ingest import chunk_text, prepare_documents

        chunks = chunk_text(text)
        if not chunks:
            logger.error("No chunks produced — skipping ingestion")
            return 0

        logger.info(f"Embedding {len(chunks)} chunks...")

        chunk_ids, metadatas = prepare_documents(chunks, metadata, doc_id_prefix)
        embeddings = await self.embedding_model.embed_batch(chunks)

        valid_documents = []
        skipped = 0

        for chunk_id, chunk_content, embedding, meta in zip(
            chunk_ids, chunks, embeddings, metadatas
        ):
            if not embedding:
                logger.warning(f"Skipping chunk '{chunk_id}' — embedding failed")
                skipped += 1
                continue

            valid_documents.append({
                "id": chunk_id,
                "embedding": embedding,
                "content": chunk_content,
                "metadata": meta,
            })

        if skipped:
            logger.warning(f"Skipped {skipped} chunks due to embedding failures")

        if not valid_documents:
            logger.error("No valid chunks to upsert")
            return 0

        logger.info(f"Upserting {len(valid_documents)} chunks into vector store...")
        success = await self.vector_store.upsert(valid_documents)

        if success:
            logger.info(f"Successfully upserted {len(valid_documents)} chunks")
            return len(valid_documents)

        logger.error("Vector store upsert failed")
        return 0

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

        Args:
            raw_results: Raw dicts from BaseVectorStore.query()
            threshold: Minimum score to include

        Returns:
            Filtered, structured list of RetrievedChunk objects
        """
        chunks = []

        for result in raw_results:
            score = result.get("score", 0.0)

            if score < threshold:
                logger.debug(
                    f"Chunk '{result.get('id')}' filtered — "
                    f"score {score:.3f} below threshold {threshold}"
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