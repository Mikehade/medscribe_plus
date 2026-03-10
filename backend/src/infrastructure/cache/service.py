import asyncio
import functools
import hashlib
import json
import logging
from typing import Any, Callable, Optional, Union

from src.infrastructure.cache.redis.manager import RedisCacheManager

from utils.logger import get_logger

logger = get_logger()


def _build_key(namespace: str, *args, **kwargs) -> str:
    """
    Build a deterministic cache key from a namespace and function arguments.
    Example: "school:get_by_id:abc123"
    """
    raw = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    fingerprint = hashlib.md5(raw.encode()).hexdigest()
    return f"{namespace}:{fingerprint}"


class CacheService:
    """
    High-level cache service with decorator support.
    Inject this wherever caching is needed — services, repositories, tools, etc.

    Usage:
        cache_service = CacheService(redis_cache_manager)

        # As a decorator on any async function
        @cache_service.cache(namespace="school", ttl=300)
        async def get_school(school_id: str): ...

        # Manually
        await cache_service.set("my:key", data, ttl=60)
        await cache_service.invalidate("my:key")
        await cache_service.invalidate_namespace("school")
    """

    def __init__(self, manager: RedisCacheManager):
        self._manager = manager

    # ------------------------------------------------------------------
    # Direct access methods
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        return await self._manager.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        await self._manager.set(key, value, ttl)

    async def invalidate(self, key: str) -> None:
        await self._manager.delete(key)

    async def invalidate_namespace(self, namespace: str) -> int:
        """Bust all keys under a namespace, e.g. 'school' clears 'school:*'"""
        return await self._manager.clear_pattern(f"{namespace}:*")

    async def exists(self, key: str) -> bool:
        return await self._manager.exists(key)

    async def get_or_set(self, key: str, value_fn: Callable, ttl: Optional[int] = None) -> Any:
        return await self._manager.get_or_set(key, value_fn, ttl)

    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        return await self._manager.increment(key, amount, ttl)

    # ------------------------------------------------------------------
    # Decorators
    # ------------------------------------------------------------------

    def cache(
        self,
        namespace: str,
        ttl: Optional[int] = 300,
        key_builder: Optional[Callable] = None,
    ):
        """
        Decorator to cache the return value of any async function.

        Args:
            namespace: Logical grouping for the cache key (e.g. "school", "program").
            ttl:       Time-to-live in seconds. None = no expiry.
            key_builder: Optional custom fn(namespace, *args, **kwargs) -> str.
                         Defaults to md5 hash of serialized args.

        Example:
            @cache_service.cache(namespace="school", ttl=600)
            async def get_school_by_id(school_id: str) -> School: ...
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                build = key_builder or _build_key
                cache_key = build(namespace, *args, **kwargs)

                cached = await self._manager.get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache HIT: {cache_key}")
                    return cached

                logger.debug(f"Cache MISS: {cache_key}")
                result = await func(*args, **kwargs)

                if result is not None:
                    await self._manager.set(cache_key, result, ttl)

                return result
            return wrapper
        return decorator

    def cache_invalidate(self, namespace: str, key_builder: Optional[Callable] = None):
        """
        Decorator that invalidates a specific cache key after the decorated
        function executes (write-through invalidation).

        Useful on update/delete methods so stale data is immediately evicted.

        Example:
            @cache_service.cache_invalidate(namespace="school")
            async def update_school(school_id: str, data: dict) -> School: ...
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                build = key_builder or _build_key
                cache_key = build(namespace, *args, **kwargs)
                await self._manager.delete(cache_key)
                logger.debug(f"Cache INVALIDATED: {cache_key}")
                return result
            return wrapper
        return decorator

    def cache_invalidate_namespace(self, namespace: str):
        """
        Decorator that busts the entire namespace after the decorated
        function executes. Use on broad mutations (bulk update, delete all, etc).

        Example:
            @cache_service.cache_invalidate_namespace("program")
            async def bulk_update_programs(data: list) -> None: ...
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                deleted = await self._manager.clear_pattern(f"{namespace}:*")
                logger.debug(f"Cache NAMESPACE BUSTED: {namespace}:* ({deleted} keys)")
                return result
            return wrapper
        return decorator

    def rate_limit(self, key: str, limit: int, window_seconds: int):
        """
        Decorator for simple fixed-window rate limiting on any async function.
        Raises RuntimeError when the limit is exceeded within the window.

        Example:
            @cache_service.rate_limit(key="ai:bedrock", limit=10, window_seconds=60)
            async def call_bedrock(...): ...
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                count = await self._manager.increment(key, ttl=window_seconds)
                if count > limit:
                    raise RuntimeError(
                        f"Rate limit exceeded for '{key}': {count}/{limit} "
                        f"within {window_seconds}s window."
                    )
                return await func(*args, **kwargs)
            return wrapper
        return decorator