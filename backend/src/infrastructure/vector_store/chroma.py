"""
ChromaDB vector store implementation.
Concrete implementation of BaseVectorStore using ChromaDB as the backend.
"""
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.infrastructure.vector_store.base import BaseVectorStore
from utils.logger import get_logger

logger = get_logger()


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation of the BaseVectorStore interface.

    Supports both local persistent storage (development) and
    HTTP client mode for a remote ChromaDB server (staging/production).

    Usage:
        # Local persistent (dev)
        store = ChromaVectorStore(
            collection_name="clinical_docs",
            persist_directory="./chroma_db"
        )

        # Remote server (prod)
        store = ChromaVectorStore(
            collection_name="clinical_docs",
            host="your-chroma-host",
            port=8000
        )

        await store.initialize()
    """

    def __init__(
        self,
        collection_name: str,
        embedding_dimension: int = 1024,
        distance_metric: str = "cosine",
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 8020,
        **kwargs
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_dimension: Must match Nova Embed output — 1024
            distance_metric: "cosine", "l2", or "ip".
                             Cosine is recommended for semantic search.
            persist_directory: Local path for persistent storage (dev/local mode).
                               If None and no host provided, uses in-memory.
            host: Remote ChromaDB server host (production mode)
            port: Remote ChromaDB server port (default: 8020)
            **kwargs: Additional ChromaDB settings
        """
        super().__init__(
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
            distance_metric=distance_metric,
        )

        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None

        # Map our metric names to ChromaDB's expected space names
        self._metric_map = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip",
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Initialize ChromaDB client and ensure the collection exists.
        Must be awaited before any read/write operations.
        """
        try:
            if self.host:
                # Remote HTTP client — for staging/production
                logger.info(f"Connecting to remote ChromaDB at {self.host}:{self.port}")
                self._client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=Settings(anonymized_telemetry=False),
                )
            elif self.persist_directory:
                # Local persistent client — for development
                logger.info(f"Initializing local ChromaDB at: {self.persist_directory}")
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                # In-memory — for testing only
                logger.warning("ChromaDB running in-memory. Data will not persist.")
                self._client = chromadb.EphemeralClient(
                    settings=Settings(anonymized_telemetry=False),
                )

            # Get or create the collection with the correct distance metric
            chroma_metric = self._metric_map.get(self.distance_metric, "cosine")
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": chroma_metric},
            )

            count = self._collection.count()
            logger.info(
                f"ChromaDB collection '{self.collection_name}' ready. "
                f"Existing chunks: {count}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            raise

    async def close(self) -> None:
        """
        Release ChromaDB client resources.
        ChromaDB PersistentClient auto-persists, so this is mostly a cleanup signal.
        """
        self._collection = None
        self._client = None
        logger.info("ChromaDB connection closed.")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def upsert(
        self,
        documents: List[Dict[str, Any]],
    ) -> bool:
        """
        Insert or update document chunks in the ChromaDB collection.

        Each document dict must contain:
            - id (str): Unique chunk identifier
            - embedding (List[float]): 1024-dim Nova embedding vector
            - content (str): Raw text of the chunk
            - metadata (Dict): Arbitrary metadata (source, doc_type, page, etc.)

        Args:
            documents: List of document chunk dicts

        Returns:
            True if upsert succeeded
        """
        self._assert_initialized()

        try:
            ids = [doc["id"] for doc in documents]
            embeddings = [doc["embedding"] for doc in documents]
            contents = [doc["content"] for doc in documents]

            # ChromaDB metadata values must be str, int, float, or bool
            # Flatten any nested metadata fields before storing
            metadatas = [
                self._sanitize_metadata(doc.get("metadata", {}))
                for doc in documents
            ]

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )

            logger.info(f"Upserted {len(documents)} chunks into '{self.collection_name}'")
            return True

        except Exception as e:
            logger.error(f"ChromaDB upsert failed: {e}", exc_info=True)
            return False

    async def delete(
        self,
        document_ids: List[str],
    ) -> bool:
        """
        Delete document chunks by their IDs.

        Args:
            document_ids: List of chunk IDs to remove

        Returns:
            True if deletion succeeded
        """
        self._assert_initialized()

        try:
            self._collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} chunks from '{self.collection_name}'")
            return True

        except Exception as e:
            logger.error(f"ChromaDB delete failed: {e}", exc_info=True)
            return False

    async def delete_collection(self) -> bool:
        """
        Drop the entire collection. Irreversible.

        Returns:
            True if the collection was deleted
        """
        self._assert_initialized()

        try:
            self._client.delete_collection(name=self.collection_name)
            self._collection = None
            logger.warning(f"Collection '{self.collection_name}' deleted.")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks for a query embedding.

        Args:
            query_embedding: 1024-dim Nova embedding of the query
            top_k: Number of results to return
            filters: Optional ChromaDB `where` filter dict
                     e.g. {"doc_type": "clinical_guideline"}
                     Supports ChromaDB operators: $eq, $ne, $in, $nin, $and, $or

        Returns:
            List of result dicts with keys: id, content, metadata, score
        """
        self._assert_initialized()

        try:
            query_params: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }

            if filters:
                query_params["where"] = filters

            results = self._collection.query(**query_params)

            # Unpack ChromaDB's nested list response into flat result dicts
            return self._unpack_query_results(results)

        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}", exc_info=True)
            return []

    async def get_by_id(
        self,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single document chunk by ID.

        Args:
            document_id: The chunk ID to fetch

        Returns:
            Document dict or None if not found
        """
        self._assert_initialized()

        try:
            result = self._collection.get(
                ids=[document_id],
                include=["documents", "metadatas", "embeddings"],
            )

            if not result["ids"] or not result["ids"][0]:
                return None

            return {
                "id": result["ids"][0],
                "content": result["documents"][0],
                "metadata": result["metadatas"][0],
                "embedding": result["embeddings"][0],
            }

        except Exception as e:
            logger.error(f"ChromaDB get_by_id failed: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    async def count(self) -> int:
        """
        Return the total number of chunks in the collection.

        Returns:
            Integer count
        """
        self._assert_initialized()

        try:
            return self._collection.count()
        except Exception as e:
            logger.error(f"ChromaDB count failed: {e}", exc_info=True)
            return 0

    async def collection_exists(self) -> bool:
        """
        Check if the collection exists in ChromaDB.

        Returns:
            True if the collection exists
        """
        if not self._client:
            return False

        try:
            existing = [c.name for c in self._client.list_collections()]
            return self.collection_name in existing
        except Exception as e:
            logger.error(f"ChromaDB collection_exists check failed: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_initialized(self) -> None:
        """
        Raise a RuntimeError if initialize() has not been called yet.
        Guards every read/write method against un-initialized state.
        """
        if self._client is None or self._collection is None:
            raise RuntimeError(
                "ChromaVectorStore is not initialized. "
                "Call `await store.initialize()` before use."
            )

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten metadata to ChromaDB-compatible scalar types.
        ChromaDB only accepts str, int, float, or bool as metadata values.
        Lists and nested dicts are JSON-serialized to strings.

        Args:
            metadata: Raw metadata dict

        Returns:
            Sanitized metadata dict safe for ChromaDB storage
        """
        import json

        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                sanitized[key] = ""
            else:
                # Serialize complex types (lists, dicts) to JSON strings
                sanitized[key] = json.dumps(value)
        return sanitized

    def _unpack_query_results(
        self,
        results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Unpack ChromaDB's nested query response into a clean flat list.

        ChromaDB returns results as lists-of-lists (one inner list per query).
        Since we always query with a single embedding, we take index [0].

        ChromaDB returns distances (lower = more similar for L2/cosine).
        We convert to a similarity score (1 - distance) for cosine so
        higher score always means more relevant — consistent with the
        contract defined in BaseVectorStore.query().

        Args:
            results: Raw ChromaDB query response

        Returns:
            Flat list of result dicts
        """
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        unpacked = []
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            # Convert distance → similarity score
            score = 1.0 - distance if self.distance_metric == "cosine" else -distance

            unpacked.append({
                "id": chunk_id,
                "content": content,
                "metadata": metadata or {},
                "score": round(score, 4),
            })

        # Sort descending by score — most relevant first
        unpacked.sort(key=lambda x: x["score"], reverse=True)

        return unpacked