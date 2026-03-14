"""
Unit tests for CacheService
"""
import pytest
from src.infrastructure.cache.service import CacheService


class TestCacheServiceFixtures:
    """Shared fixtures for CacheService tests"""

    @pytest.fixture
    def mock_manager(self, mocker):
        """Create a mock RedisCacheManager"""
        manager = mocker.AsyncMock()
        manager.get = mocker.AsyncMock(return_value=None)
        manager.set = mocker.AsyncMock()
        manager.delete = mocker.AsyncMock()
        manager.exists = mocker.AsyncMock(return_value=False)
        manager.clear_pattern = mocker.AsyncMock(return_value=0)
        manager.get_or_set = mocker.AsyncMock()
        manager.increment = mocker.AsyncMock(return_value=1)
        return manager

    @pytest.fixture
    def cache_service(self, mock_manager):
        """Create CacheService with mocked manager"""
        return CacheService(manager=mock_manager)


class TestCacheServiceDirectMethods(TestCacheServiceFixtures):
    """Tests for direct access methods: get, set, invalidate, etc."""

    @pytest.mark.asyncio
    async def test_get_delegates_to_manager(self, cache_service, mock_manager):
        """get() should delegate directly to manager.get()"""
        mock_manager.get.return_value = {"data": 1}
        result = await cache_service.get("my:key")
        mock_manager.get.assert_called_once_with("my:key")
        assert result == {"data": 1}

    @pytest.mark.asyncio
    async def test_set_delegates_to_manager(self, cache_service, mock_manager):
        """set() should delegate to manager.set() with key, value, and ttl"""
        await cache_service.set("my:key", {"data": 1}, ttl=60)
        mock_manager.set.assert_called_once_with("my:key", {"data": 1}, 60)

    @pytest.mark.asyncio
    async def test_invalidate_calls_manager_delete(self, cache_service, mock_manager):
        """invalidate() should call manager.delete() with the key"""
        await cache_service.invalidate("my:key")
        mock_manager.delete.assert_called_once_with("my:key")

    @pytest.mark.asyncio
    async def test_invalidate_namespace_calls_clear_pattern(self, cache_service, mock_manager):
        """invalidate_namespace() should call manager.clear_pattern() with namespace wildcard"""
        mock_manager.clear_pattern.return_value = 5
        result = await cache_service.invalidate_namespace("school")
        mock_manager.clear_pattern.assert_called_once_with("school:*")
        assert result == 5

    @pytest.mark.asyncio
    async def test_exists_delegates_to_manager(self, cache_service, mock_manager):
        """exists() should delegate to manager.exists()"""
        mock_manager.exists.return_value = True
        result = await cache_service.exists("my:key")
        mock_manager.exists.assert_called_once_with("my:key")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_or_set_delegates_to_manager(self, cache_service, mock_manager, mocker):
        """get_or_set() should delegate to manager.get_or_set()"""
        value_fn = mocker.AsyncMock(return_value="data")
        mock_manager.get_or_set.return_value = "data"

        result = await cache_service.get_or_set("key", value_fn, ttl=300)

        mock_manager.get_or_set.assert_called_once_with("key", value_fn, 300)
        assert result == "data"

    @pytest.mark.asyncio
    async def test_increment_delegates_to_manager(self, cache_service, mock_manager):
        """increment() should delegate to manager.increment()"""
        mock_manager.increment.return_value = 3
        result = await cache_service.increment("counter", amount=2, ttl=60)
        mock_manager.increment.assert_called_once_with("counter", 2, 60)
        assert result == 3


class TestCacheServiceCacheDecorator(TestCacheServiceFixtures):
    """Tests for the @cache decorator"""

    @pytest.mark.asyncio
    async def test_cache_decorator_returns_cached_value_on_hit(
        self, cache_service, mock_manager
    ):
        """Should return cached value without calling the wrapped function"""
        mock_manager.get.return_value = {"school": "cached"}

        @cache_service.cache(namespace="school", ttl=300)
        async def get_school(school_id: str):
            return {"school": "fresh"}

        result = await get_school("abc123")

        assert result == {"school": "cached"}
        mock_manager.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_decorator_calls_fn_and_caches_on_miss(
        self, cache_service, mock_manager
    ):
        """Should call wrapped function and cache result on cache miss"""
        mock_manager.get.return_value = None

        @cache_service.cache(namespace="school", ttl=300)
        async def get_school(school_id: str):
            return {"school": "fresh"}

        result = await get_school("abc123")

        assert result == {"school": "fresh"}
        mock_manager.set.assert_called_once()
        set_args = mock_manager.set.call_args
        assert set_args[0][2] == 300  # TTL passed correctly

    @pytest.mark.asyncio
    async def test_cache_decorator_does_not_cache_none_result(
        self, cache_service, mock_manager
    ):
        """Should not cache when wrapped function returns None"""
        mock_manager.get.return_value = None

        @cache_service.cache(namespace="school", ttl=300)
        async def get_school(school_id: str):
            return None

        result = await get_school("abc123")

        assert result is None
        mock_manager.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_decorator_uses_namespace_in_key(
        self, cache_service, mock_manager
    ):
        """Cache key should be prefixed with the given namespace"""
        mock_manager.get.return_value = None

        @cache_service.cache(namespace="program", ttl=60)
        async def get_program(program_id: str):
            return {"program": "data"}

        await get_program("xyz")

        get_key = mock_manager.get.call_args[0][0]
        assert get_key.startswith("program:")

    @pytest.mark.asyncio
    async def test_cache_decorator_same_args_produce_same_key(
        self, cache_service, mock_manager
    ):
        """Same arguments should always produce the same cache key"""
        mock_manager.get.return_value = None

        @cache_service.cache(namespace="school", ttl=300)
        async def get_school(school_id: str):
            return {"school": "data"}

        await get_school("abc123")
        first_key = mock_manager.get.call_args[0][0]

        mock_manager.get.reset_mock()
        await get_school("abc123")
        second_key = mock_manager.get.call_args[0][0]

        assert first_key == second_key

    @pytest.mark.asyncio
    async def test_cache_decorator_different_args_produce_different_keys(
        self, cache_service, mock_manager
    ):
        """Different arguments should produce different cache keys"""
        mock_manager.get.return_value = None

        @cache_service.cache(namespace="school", ttl=300)
        async def get_school(school_id: str):
            return {"school": "data"}

        await get_school("abc123")
        key_1 = mock_manager.get.call_args[0][0]

        mock_manager.get.reset_mock()
        await get_school("xyz789")
        key_2 = mock_manager.get.call_args[0][0]

        assert key_1 != key_2

    @pytest.mark.asyncio
    async def test_cache_decorator_uses_custom_key_builder(
        self, cache_service, mock_manager, mocker
    ):
        """Should use custom key_builder when provided"""
        mock_manager.get.return_value = None
        custom_builder = mocker.Mock(return_value="custom:key:123")

        @cache_service.cache(namespace="school", ttl=300, key_builder=custom_builder)
        async def get_school(school_id: str):
            return {"school": "data"}

        await get_school("abc")

        custom_builder.assert_called_once()
        mock_manager.get.assert_called_once_with("custom:key:123")

    @pytest.mark.asyncio
    async def test_cache_decorator_preserves_function_name(self, cache_service):
        """Decorator should preserve the wrapped function's __name__"""
        @cache_service.cache(namespace="school", ttl=300)
        async def get_school(school_id: str):
            return {}

        assert get_school.__name__ == "get_school"


class TestCacheServiceCacheInvalidateDecorator(TestCacheServiceFixtures):
    """Tests for the @cache_invalidate decorator"""

    @pytest.mark.asyncio
    async def test_cache_invalidate_executes_function(
        self, cache_service, mock_manager, mocker
    ):
        """Should execute the decorated function and return its result"""
        @cache_service.cache_invalidate(namespace="school")
        async def update_school(school_id: str, data: dict):
            return {"updated": True}

        result = await update_school("abc123", {"name": "New Name"})
        assert result == {"updated": True}

    @pytest.mark.asyncio
    async def test_cache_invalidate_deletes_key_after_execution(
        self, cache_service, mock_manager
    ):
        """Should delete the cache key after the function runs"""
        @cache_service.cache_invalidate(namespace="school")
        async def update_school(school_id: str):
            return {"updated": True}

        await update_school("abc123")

        mock_manager.delete.assert_called_once()
        deleted_key = mock_manager.delete.call_args[0][0]
        assert deleted_key.startswith("school:")

    @pytest.mark.asyncio
    async def test_cache_invalidate_uses_custom_key_builder(
        self, cache_service, mock_manager, mocker
    ):
        """Should use custom key_builder when provided"""
        custom_builder = mocker.Mock(return_value="school:fixed:key")

        @cache_service.cache_invalidate(namespace="school", key_builder=custom_builder)
        async def update_school(school_id: str):
            return {}

        await update_school("abc")

        mock_manager.delete.assert_called_once_with("school:fixed:key")

    @pytest.mark.asyncio
    async def test_cache_invalidate_preserves_function_name(self, cache_service):
        """Decorator should preserve the wrapped function's __name__"""
        @cache_service.cache_invalidate(namespace="school")
        async def update_school(school_id: str):
            return {}

        assert update_school.__name__ == "update_school"


class TestCacheServiceCacheInvalidateNamespaceDecorator(TestCacheServiceFixtures):
    """Tests for the @cache_invalidate_namespace decorator"""

    @pytest.mark.asyncio
    async def test_cache_invalidate_namespace_executes_function(
        self, cache_service, mock_manager
    ):
        """Should execute the decorated function and return its result"""
        @cache_service.cache_invalidate_namespace("program")
        async def bulk_update(data: list):
            return {"count": len(data)}

        result = await bulk_update([1, 2, 3])
        assert result == {"count": 3}

    @pytest.mark.asyncio
    async def test_cache_invalidate_namespace_busts_all_keys(
        self, cache_service, mock_manager
    ):
        """Should call clear_pattern with the namespace wildcard"""
        mock_manager.clear_pattern.return_value = 7

        @cache_service.cache_invalidate_namespace("program")
        async def bulk_update(data: list):
            return {}

        await bulk_update([])

        mock_manager.clear_pattern.assert_called_once_with("program:*")

    @pytest.mark.asyncio
    async def test_cache_invalidate_namespace_preserves_function_name(self, cache_service):
        """Decorator should preserve the wrapped function's __name__"""
        @cache_service.cache_invalidate_namespace("program")
        async def bulk_update_programs(data: list):
            return {}

        assert bulk_update_programs.__name__ == "bulk_update_programs"


class TestCacheServiceRateLimitDecorator(TestCacheServiceFixtures):
    """Tests for the @rate_limit decorator"""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_call_within_limit(
        self, cache_service, mock_manager
    ):
        """Should execute the function when count is within limit"""
        mock_manager.increment.return_value = 5  # within limit of 10

        @cache_service.rate_limit(key="ai:bedrock", limit=10, window_seconds=60)
        async def call_bedrock():
            return "response"

        result = await call_bedrock()
        assert result == "response"

    @pytest.mark.asyncio
    async def test_rate_limit_raises_when_limit_exceeded(
        self, cache_service, mock_manager
    ):
        """Should raise RuntimeError when count exceeds limit"""
        mock_manager.increment.return_value = 11  # over limit of 10

        @cache_service.rate_limit(key="ai:bedrock", limit=10, window_seconds=60)
        async def call_bedrock():
            return "response"

        with pytest.raises(RuntimeError) as exc_info:
            await call_bedrock()

        assert "Rate limit exceeded" in str(exc_info.value)
        assert "ai:bedrock" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rate_limit_increments_counter_with_window(
        self, cache_service, mock_manager
    ):
        """Should call increment with the correct key and window TTL"""
        mock_manager.increment.return_value = 1

        @cache_service.rate_limit(key="my:endpoint", limit=5, window_seconds=30)
        async def my_fn():
            return "ok"

        await my_fn()

        mock_manager.increment.assert_called_once_with("my:endpoint", ttl=30)

    @pytest.mark.asyncio
    async def test_rate_limit_allows_call_exactly_at_limit(
        self, cache_service, mock_manager
    ):
        """Should allow call when count equals the limit exactly"""
        mock_manager.increment.return_value = 10  # exactly at limit of 10

        @cache_service.rate_limit(key="ai:bedrock", limit=10, window_seconds=60)
        async def call_bedrock():
            return "response"

        result = await call_bedrock()
        assert result == "response"

    @pytest.mark.asyncio
    async def test_rate_limit_preserves_function_name(self, cache_service):
        """Decorator should preserve the wrapped function's __name__"""
        @cache_service.rate_limit(key="key", limit=10, window_seconds=60)
        async def call_service():
            return {}

        assert call_service.__name__ == "call_service"