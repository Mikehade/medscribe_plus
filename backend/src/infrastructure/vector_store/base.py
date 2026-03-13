
"""
Base abstract class for vector store providers.
Defines the common interface that all vector store implementations must follow.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger()


class BaseVectorStore(ABC):
    """
    Abstract base class for all vector store providers.

    Provides a unified interface for document storage and similarity
    retrieval, allowing the RAG service to remain provider-agnostic.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_dimension: int = 1024,
        distance_metric: str = "cosine",
        **kwargs
    ):
        """
        Initialize base vector store.

        Args:
            collection_name: Name of the collection / index to operate on
            embedding_dimension: Dimensionality of the embedding vectors.
                                 Amazon Nova embed produces 1024-dim vectors.
            distance_metric: Similarity metric — "cosine", "l2", or "ip" (inner product)
            **kwargs: Additional provider-specific parameters
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the vector store connection and ensure the collection exists.
        Must be called once before any read/write operations.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Gracefully close the vector store connection.
        """
        pass

    @abstractmethod
    async def upsert(
        self,
        documents: List[Dict[str, Any]],
    ) -> bool:
        """
        Insert or update documents in the vector store.

        Each document dict must contain:
            - id (str): Unique identifier for the document chunk
            - embedding (List[float]): Pre-computed embedding vector
            - content (str): Raw text content of the chunk
            - metadata (Dict): Arbitrary metadata (source, page, doc_type, etc.)

        Args:
            documents: List of document dicts as described above

        Returns:
            True if all documents were upserted successfully
        """
        pass

    @abstractmethod
    async def delete(
        self,
        document_ids: List[str],
    ) -> bool:
        """
        Delete documents from the vector store by their IDs.

        Args:
            document_ids: List of document chunk IDs to delete

        Returns:
            True if deletion was successful
        """
        pass

    # @abstractmethod
    # async def delete_collection(self) -> bool:
    #     """
    #     Drop the entire collection from the vector store.
    #     Use with caution — this is irreversible.

    #     Returns:
    #         True if the collection was dropped successfully
    #     """
    #     pass

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def query( 
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar document chunks for a query embedding.

        Args:
            query_embedding: The embedded query vector to search against
            top_k: Number of results to return
            filters: Optional metadata filters to narrow the search space
                     e.g. {"doc_type": "clinical_guideline", "specialty": "cardiology"}

        Returns:
            List of result dicts, each containing:
                - id (str): Document chunk ID
                - content (str): Raw text of the chunk
                - metadata (Dict): Chunk metadata
                - score (float): Similarity score (higher = more similar)
        """
        pass

    @abstractmethod
    async def get_by_id(
        self,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single document chunk by its ID.

        Args:
            document_id: The chunk ID to fetch

        Returns:
            Document dict or None if not found
        """
        pass

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @abstractmethod
    async def count(self) -> int:
        """
        Return the total number of document chunks in the collection.

        Returns:
            Integer count of stored chunks
        """
        pass

    @abstractmethod
    async def collection_exists(self) -> bool:
        """
        Check whether the collection already exists in the vector store.

        Returns:
            True if the collection exists
        """
        pass