"""
Abstract base class for embedding model providers.

Defines the minimal interface all embedding implementations must satisfy.
Unlike LLM models, embedding models have a simpler contract — no prompting,
streaming, or tool calling. Just: text in, vector out.
"""
from abc import ABC, abstractmethod
from typing import List

from utils.logger import get_logger

logger = get_logger()


class BaseEmbeddingModel(ABC):
    """
    Abstract base for embedding model providers.

    Concrete implementations (Bedrock Nova, HuggingFace, OpenAI, etc.)
    inherit this class and implement the two abstract methods below.

    The interface is intentionally minimal:
        - embed_text   : single string → dense vector
        - embed_batch  : list of strings → list of dense vectors

    All provider-specific concerns (auth, retry, dimensionality) live
    in the concrete subclass. Consumers (RAGService) depend only on this
    abstract interface, making providers fully swappable.
    """

    def __init__(self, model_id: str, **kwargs):
        """
        Args:
            model_id: Provider-specific model identifier
            **kwargs: Additional provider-specific parameters
        """
        self.model_id = model_id

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string into a dense vector.

        Args:
            text: The input text to embed

        Returns:
            A list of floats representing the embedding vector.
            Returns an empty list on failure — never raises to callers.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts. Each text is embedded independently.

        Implementations may call embed_text in a loop (most providers
        do not support true batch embedding in a single API call).

        Args:
            texts: List of input strings

        Returns:
            List of embedding vectors in the same order as input.
            Failed embeddings are returned as empty lists — callers
            should validate before storing.
        """
        pass

    @property
    def embedding_dim(self) -> int:
        """
        Return the output dimensionality of this model's vectors.

        Subclasses should override this with their actual dimension.
        Used by callers that need to configure vector store schema ahead
        of embedding (e.g. ChromaDB collection creation).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must define embedding_dim"
        )