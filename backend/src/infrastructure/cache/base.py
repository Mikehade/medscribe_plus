from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseCache(ABC):
    """
    Abstract cache interface.
    Any cache backend (Redis, Memcached, in-memory) must implement this contract.
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def exists(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching a glob pattern. Returns count of deleted keys."""
        raise NotImplementedError

    @abstractmethod
    async def set_ttl(self, key: str, ttl: int) -> None:
        """Update TTL on an existing key."""
        raise NotImplementedError