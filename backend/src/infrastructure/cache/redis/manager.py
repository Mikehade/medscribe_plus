import json
import logging
from typing import Any, Optional

from src.infrastructure.cache.base import BaseCache
from src.infrastructure.cache.redis.client import RedisClient
from utils.logger import get_logger

logger = get_logger()


class RedisCacheManager(BaseCache):
    """
    Redis-backed cache implementation.
    Handles serialization/deserialization and delegates all I/O to RedisClient.
    """

    def __init__(self, client: RedisClient):
        self._client = client

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        try:
            raw = await self._client.client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Cache GET error for key '{key}': {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            serialized = json.dumps(value)
            if ttl:
                await self._client.client.setex(key, ttl, serialized)
            else:
                await self._client.client.set(key, serialized)
        except Exception as e:
            logger.error(f"Cache SET error for key '{key}': {e}")

    async def delete(self, key: str) -> None:
        try:
            await self._client.client.delete(key)
        except Exception as e:
            logger.error(f"Cache DELETE error for key '{key}': {e}")

    async def exists(self, key: str) -> bool:
        try:
            return bool(await self._client.client.exists(key))
        except Exception as e:
            logger.error(f"Cache EXISTS error for key '{key}': {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern, e.g. 'school:*' or 'program:list:*'"""
        try:
            keys = await self._client.client.keys(pattern)
            if not keys:
                return 0
            return await self._client.client.delete(*keys)
        except Exception as e:
            logger.error(f"Cache CLEAR_PATTERN error for pattern '{pattern}': {e}")
            return 0

    async def set_ttl(self, key: str, ttl: int) -> None:
        try:
            await self._client.client.expire(key, ttl)
        except Exception as e:
            logger.error(f"Cache SET_TTL error for key '{key}': {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def get_or_set(self, key: str, value_fn, ttl: Optional[int] = None) -> Any:
        """
        Return cached value if it exists, otherwise call value_fn(),
        cache the result, and return it.

        value_fn can be sync or async.
        """
        cached = await self.get(key)
        if cached is not None:
            return cached

        import asyncio
        value = await value_fn() if asyncio.iscoroutinefunction(value_fn) else value_fn()
        await self.set(key, value, ttl)
        return value

    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Atomic increment — useful for rate limiting and counters."""
        try:
            result = await self._client.client.incrby(key, amount)
            if ttl and result == amount:  # first increment — set TTL
                await self.set_ttl(key, ttl)
            return result
        except Exception as e:
            logger.error(f"Cache INCREMENT error for key '{key}': {e}")
            return 0
