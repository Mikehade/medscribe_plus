"""
Unit tests for RedisCacheManager
"""
import json
import pytest
from src.infrastructure.cache.redis.manager import RedisCacheManager


class TestRedisCacheManagerFixtures:
    """Shared fixtures for RedisCacheManager tests"""

    @pytest.fixture
    def mock_redis_client(self, mocker):
        """Create a mock Redis client with all Redis commands mocked"""
        client = mocker.AsyncMock()
        client.get = mocker.AsyncMock(return_value=None)
        client.set = mocker.AsyncMock(return_value=True)
        client.setex = mocker.AsyncMock(return_value=True)
        client.delete = mocker.AsyncMock(return_value=1)
        client.exists = mocker.AsyncMock(return_value=0)
        client.keys = mocker.AsyncMock(return_value=[])
        client.expire = mocker.AsyncMock(return_value=True)
        client.incrby = mocker.AsyncMock(return_value=1)
        return client

    @pytest.fixture
    def mock_redis_client_wrapper(self, mocker, mock_redis_client):
        """Create a mock RedisClient wrapper whose .client is the raw mock"""
        wrapper = mocker.Mock()
        wrapper.client = mock_redis_client
        return wrapper

    @pytest.fixture
    def manager(self, mock_redis_client_wrapper):
        """Create RedisCacheManager with mocked client"""
        return RedisCacheManager(client=mock_redis_client_wrapper)


class TestRedisCacheManagerGet(TestRedisCacheManagerFixtures):
    """Tests for get()"""

    @pytest.mark.asyncio
    async def test_get_returns_deserialized_value(self, manager, mock_redis_client):
        """Should deserialize JSON string from Redis and return Python object"""
        # Arrange
        mock_redis_client.get.return_value = json.dumps({"name": "Test School"})

        # Act
        result = await manager.get("school:123")

        # Assert
        assert result == {"name": "Test School"}
        mock_redis_client.get.assert_called_once_with("school:123")

    @pytest.mark.asyncio
    async def test_get_returns_none_when_key_missing(self, manager, mock_redis_client):
        """Should return None when key does not exist in Redis"""
        mock_redis_client.get.return_value = None
        result = await manager.get("missing:key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_none_on_redis_exception(self, manager, mock_redis_client):
        """Should return None and not raise when Redis throws an exception"""
        mock_redis_client.get.side_effect = Exception("Redis connection lost")
        result = await manager.get("any:key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_handles_list_value(self, manager, mock_redis_client):
        """Should correctly deserialize a cached list"""
        mock_redis_client.get.return_value = json.dumps([1, 2, 3])
        result = await manager.get("list:key")
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_get_handles_primitive_values(self, manager, mock_redis_client):
        """Should correctly deserialize primitive types (int, float, bool, str)"""
        for value in [42, 3.14, True, "hello"]:
            mock_redis_client.get.return_value = json.dumps(value)
            result = await manager.get("key")
            assert result == value


class TestRedisCacheManagerSet(TestRedisCacheManagerFixtures):
    """Tests for set()"""

    @pytest.mark.asyncio
    async def test_set_without_ttl_calls_redis_set(self, manager, mock_redis_client):
        """Should call redis set (not setex) when no TTL is given"""
        await manager.set("key:1", {"data": "value"})

        mock_redis_client.set.assert_called_once_with("key:1", json.dumps({"data": "value"}))
        mock_redis_client.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_with_ttl_calls_redis_setex(self, manager, mock_redis_client):
        """Should call redis setex with TTL when TTL is provided"""
        await manager.set("key:1", {"data": "value"}, ttl=300)

        mock_redis_client.setex.assert_called_once_with("key:1", 300, json.dumps({"data": "value"}))
        mock_redis_client.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_serializes_complex_objects(self, manager, mock_redis_client):
        """Should JSON serialize complex nested objects"""
        data = {"nested": {"list": [1, 2, 3], "flag": True}}
        await manager.set("complex:key", data)

        call_args = mock_redis_client.set.call_args[0]
        assert json.loads(call_args[1]) == data

    @pytest.mark.asyncio
    async def test_set_does_not_raise_on_redis_exception(self, manager, mock_redis_client):
        """Should silently handle Redis exceptions during set"""
        mock_redis_client.set.side_effect = Exception("Write failed")

        # Should not raise
        await manager.set("key", "value")

    @pytest.mark.asyncio
    async def test_set_with_zero_ttl_does_not_use_setex(self, manager, mock_redis_client):
        """TTL of 0 is falsy — should use plain set not setex"""
        await manager.set("key", "value", ttl=0)

        mock_redis_client.set.assert_called_once()
        mock_redis_client.setex.assert_not_called()


class TestRedisCacheManagerDelete(TestRedisCacheManagerFixtures):
    """Tests for delete()"""

    @pytest.mark.asyncio
    async def test_delete_calls_redis_delete(self, manager, mock_redis_client):
        """Should call redis delete with the correct key"""
        await manager.delete("school:123")
        mock_redis_client.delete.assert_called_once_with("school:123")

    @pytest.mark.asyncio
    async def test_delete_does_not_raise_on_redis_exception(self, manager, mock_redis_client):
        """Should silently handle Redis exceptions during delete"""
        mock_redis_client.delete.side_effect = Exception("Delete failed")
        await manager.delete("key")  # Should not raise


class TestRedisCacheManagerExists(TestRedisCacheManagerFixtures):
    """Tests for exists()"""

    @pytest.mark.asyncio
    async def test_exists_returns_true_when_key_present(self, manager, mock_redis_client):
        """Should return True when Redis reports key exists"""
        mock_redis_client.exists.return_value = 1
        result = await manager.exists("school:123")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_when_key_absent(self, manager, mock_redis_client):
        """Should return False when Redis reports key does not exist"""
        mock_redis_client.exists.return_value = 0
        result = await manager.exists("missing:key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_returns_false_on_redis_exception(self, manager, mock_redis_client):
        """Should return False and not raise on Redis exception"""
        mock_redis_client.exists.side_effect = Exception("Redis error")
        result = await manager.exists("key")
        assert result is False


class TestRedisCacheManagerClearPattern(TestRedisCacheManagerFixtures):
    """Tests for clear_pattern()"""

    @pytest.mark.asyncio
    async def test_clear_pattern_deletes_matching_keys(self, manager, mock_redis_client):
        """Should delete all keys matching the pattern and return count"""
        mock_redis_client.keys.return_value = ["school:1", "school:2", "school:3"]
        mock_redis_client.delete.return_value = 3

        result = await manager.clear_pattern("school:*")

        mock_redis_client.keys.assert_called_once_with("school:*")
        mock_redis_client.delete.assert_called_once_with("school:1", "school:2", "school:3")
        assert result == 3

    @pytest.mark.asyncio
    async def test_clear_pattern_returns_zero_when_no_keys_match(self, manager, mock_redis_client):
        """Should return 0 without calling delete when no keys match"""
        mock_redis_client.keys.return_value = []

        result = await manager.clear_pattern("nonexistent:*")

        mock_redis_client.delete.assert_not_called()
        assert result == 0

    @pytest.mark.asyncio
    async def test_clear_pattern_returns_zero_on_redis_exception(self, manager, mock_redis_client):
        """Should return 0 and not raise on Redis exception"""
        mock_redis_client.keys.side_effect = Exception("Redis error")
        result = await manager.clear_pattern("school:*")
        assert result == 0


class TestRedisCacheManagerSetTtl(TestRedisCacheManagerFixtures):
    """Tests for set_ttl()"""

    @pytest.mark.asyncio
    async def test_set_ttl_calls_redis_expire(self, manager, mock_redis_client):
        """Should call redis expire with correct key and TTL"""
        await manager.set_ttl("school:123", 600)
        mock_redis_client.expire.assert_called_once_with("school:123", 600)

    @pytest.mark.asyncio
    async def test_set_ttl_does_not_raise_on_exception(self, manager, mock_redis_client):
        """Should silently handle exceptions during set_ttl"""
        mock_redis_client.expire.side_effect = Exception("Expire failed")
        await manager.set_ttl("key", 300)  # Should not raise


class TestRedisCacheManagerGetOrSet(TestRedisCacheManagerFixtures):
    """Tests for get_or_set()"""

    @pytest.mark.asyncio
    async def test_get_or_set_returns_cached_value_without_calling_fn(
        self, manager, mock_redis_client, mocker
    ):
        """Should return cached value immediately without calling value_fn"""
        mock_redis_client.get.return_value = json.dumps({"cached": True})
        value_fn = mocker.AsyncMock(return_value={"fresh": True})

        result = await manager.get_or_set("key", value_fn, ttl=300)

        assert result == {"cached": True}
        value_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_set_calls_sync_fn_on_cache_miss(
        self, manager, mock_redis_client
    ):
        """Should call sync value_fn and cache result on cache miss"""
        mock_redis_client.get.return_value = None
        value_fn = lambda: {"sync": True}

        result = await manager.get_or_set("key", value_fn)

        assert result == {"sync": True}

    @pytest.mark.asyncio
    async def test_get_or_set_sets_value_with_correct_ttl(
        self, manager, mock_redis_client, mocker
    ):
        """Should pass TTL to set when caching the fresh value"""
        mock_redis_client.get.return_value = None
        value_fn = mocker.AsyncMock(return_value="data")

        await manager.get_or_set("key", value_fn, ttl=120)

        mock_redis_client.setex.assert_called_once_with("key", 120, json.dumps("data"))


class TestRedisCacheManagerIncrement(TestRedisCacheManagerFixtures):
    """Tests for increment()"""

    @pytest.mark.asyncio
    async def test_increment_returns_new_count(self, manager, mock_redis_client):
        """Should return the incremented value from Redis"""
        mock_redis_client.incrby.return_value = 5
        result = await manager.increment("counter:key", amount=1)
        assert result == 5

    @pytest.mark.asyncio
    async def test_increment_sets_ttl_on_first_call(self, manager, mock_redis_client):
        """Should set TTL only on the first increment (when result equals amount)"""
        mock_redis_client.incrby.return_value = 1  # first increment
        await manager.increment("rate:key", amount=1, ttl=60)
        mock_redis_client.expire.assert_called_once_with("rate:key", 60)

    @pytest.mark.asyncio
    async def test_increment_does_not_reset_ttl_on_subsequent_calls(
        self, manager, mock_redis_client
    ):
        """Should not reset TTL when counter is already above 1"""
        mock_redis_client.incrby.return_value = 5  # not the first increment
        await manager.increment("rate:key", amount=1, ttl=60)
        mock_redis_client.expire.assert_not_called()

    @pytest.mark.asyncio
    async def test_increment_returns_zero_on_exception(self, manager, mock_redis_client):
        """Should return 0 and not raise on Redis exception"""
        mock_redis_client.incrby.side_effect = Exception("Redis error")
        result = await manager.increment("key")
        assert result == 0

    @pytest.mark.asyncio
    async def test_increment_uses_custom_amount(self, manager, mock_redis_client):
        """Should pass custom amount to incrby"""
        mock_redis_client.incrby.return_value = 10
        await manager.increment("key", amount=5)
        mock_redis_client.incrby.assert_called_once_with("key", 5)