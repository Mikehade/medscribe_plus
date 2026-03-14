from __future__ import annotations
import redis.asyncio as aioredis
from redis.asyncio import Redis
from typing import Optional
from urllib.parse import quote


class RedisClient:
    """
    Thin async Redis client wrapper.
    Manages connection lifecycle and exposes the raw client for infrastructure use.

    Tries to connect using the provided URL first. If that fails, it encodes
    the password and rebuilds the URL from individual components before retrying.
    """

    def __init__(
        self,
        url: str,
        max_connections: int = 10,
        decode_responses: bool = True,
        # Individual components for fallback URL construction
        redis_name: Optional[str] = None,
        redis_password: Optional[str] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_db: Optional[int] = None,
    ):
        self._url = url
        self._max_connections = max_connections
        self._decode_responses = decode_responses

        # Fallback connection components
        self._redis_name = redis_name
        self._redis_password = redis_password
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_db = redis_db

        self._client: Optional[Redis] = None

    def _build_encoded_url(self) -> Optional[str]:
        """Build a Redis URL with a URL-encoded password from individual components."""
        if not all([
            self._redis_name,
            self._redis_password,
            self._redis_host,
            self._redis_port is not None,
            self._redis_db is not None,
        ]):
            return None

        encoded_password = quote(self._redis_password, safe='')
        return (
            f"redis://{self._redis_name}:{encoded_password}"
            f"@{self._redis_host}:{self._redis_port}/{self._redis_db}"
        )

    async def _create_client(self, url: str) -> Redis:
        """Attempt to create and verify a Redis client from a given URL."""
        client = await aioredis.from_url(
            url,
            max_connections=self._max_connections,
            decode_responses=self._decode_responses,
        )
        await client.ping()  # Verify the connection is actually alive
        return client

    async def connect(self) -> None:
        # First attempt: use the URL as-is
        try:
            self._client = await self._create_client(self._url)
            return
        except Exception as primary_error:
            print(
                f"[RedisClient] Primary URL connection failed: {primary_error}. "
                "Attempting fallback with encoded password..."
            )

        # Second attempt: rebuild URL with URL-encoded password
        fallback_url = self._build_encoded_url()
        if not fallback_url:
            raise RuntimeError(
                "Primary Redis connection failed and insufficient credentials "
                "were provided to build a fallback URL. "
                "Pass redis_name, redis_password, redis_host, redis_port, and redis_db."
            )

        try:
            self._client = await self._create_client(fallback_url)
            self._url = fallback_url  # Persist the working URL for reconnects
        except Exception as fallback_error:
            raise RuntimeError(
                f"Both Redis connection attempts failed.\n"
                # f"  Primary error:  {primary_error}\n"
                f"  Fallback error: {fallback_error}"
            ) from fallback_error

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> Redis:
        if not self._client:
            raise RuntimeError("RedisClient is not connected. Call connect() first.")
        return self._client

    async def ping(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False