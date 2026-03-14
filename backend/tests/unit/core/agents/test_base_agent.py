"""
Unit tests for BaseAgent.

Covers:
- Cannot instantiate BaseAgent directly (abstract)
- Concrete subclass must implement process_message
- ensure_async_generator: async generator input, coroutine input,
  coroutine-returning-async-generator, list/tuple/generator input,
  unsupported type raises TypeError
"""
import asyncio
import pytest
from src.core.agents.base import BaseAgent


# ── Concrete stub ─────────────────────────────────────────────────────────────

class ConcreteAgent(BaseAgent):
    """Minimal concrete implementation used across all tests."""

    async def process_message(self, message: str, bot: str = "scribe",
                               use_history: bool = True, enable_tools: bool = True,
                               stream: bool = False):
        return {"message": message}


@pytest.fixture
def agent():
    return ConcreteAgent()


# ── Initialization / abstract enforcement ─────────────────────────────────────

class TestBaseAgentAbstract:
    """BaseAgent enforces the abstract contract correctly."""

    def test_cannot_instantiate_base_agent_directly(self):
        # Act / Assert
        with pytest.raises(TypeError):
            BaseAgent()

    def test_concrete_subclass_without_process_message_raises(self):
        # Arrange
        class Incomplete(BaseAgent):
            pass

        # Act / Assert
        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_with_process_message_instantiates(self, agent):
        # Assert
        assert agent is not None


# ── ensure_async_generator: async generator input ────────────────────────────

class TestEnsureAsyncGeneratorFromAsyncGenerator:
    """When result is already an async generator it is consumed and re-yielded."""

    @pytest.mark.asyncio
    async def test_yields_all_items_from_async_generator(self, agent):
        # Arrange
        async def source():
            yield 1
            yield 2
            yield 3

        # Act
        gen = await agent.ensure_async_generator(source())
        items = [item async for item in gen]

        # Assert
        assert items == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_returns_empty_from_empty_async_generator(self, agent):
        # Arrange
        async def source():
            return
            yield

        # Act
        gen = await agent.ensure_async_generator(source())
        items = [item async for item in gen]

        # Assert
        assert items == []

    @pytest.mark.asyncio
    async def test_preserves_item_types_from_async_generator(self, agent):
        # Arrange
        async def source():
            yield {"key": "value"}
            yield [1, 2, 3]
            yield "text"

        # Act
        gen = await agent.ensure_async_generator(source())
        items = [item async for item in gen]

        # Assert
        assert items == [{"key": "value"}, [1, 2, 3], "text"]



# ── ensure_async_generator: coroutine input ───────────────────────────────────

class TestEnsureAsyncGeneratorFromCoroutine:
    """When result is a coroutine it is awaited then wrapped."""

    @pytest.mark.asyncio
    async def test_wraps_coroutine_returning_list(self, agent):
        # Arrange
        async def coro():
            return [10, 20, 30]

        # Act
        gen = await agent.ensure_async_generator(coro())
        items = [item async for item in gen]

        # Assert
        assert items == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_wraps_coroutine_returning_tuple(self, agent):
        # Arrange
        async def coro():
            return (4, 5, 6)

        # Act
        gen = await agent.ensure_async_generator(coro())
        items = [item async for item in gen]

        # Assert
        assert items == [4, 5, 6]

    @pytest.mark.asyncio
    async def test_handles_coroutine_returning_async_generator(self, agent):
        # Arrange — coroutine that itself returns an async generator
        async def inner():
            yield "a"
            yield "b"

        async def coro():
            return inner()

        # Act
        gen = await agent.ensure_async_generator(coro())
        items = [item async for item in gen]

        # Assert
        assert items == ["a", "b"]

    @pytest.mark.asyncio
    async def test_raises_type_error_for_coroutine_returning_int(self, agent):
        # Arrange
        async def coro():
            return 42  # int is not iterable

        # Act / Assert
        with pytest.raises(TypeError):
            gen = await agent.ensure_async_generator(coro())
            [item async for item in gen]


# ── ensure_async_generator: sync iterable input ──────────────────────────────

class TestEnsureAsyncGeneratorFromSyncIterable:
    """Lists, tuples, and sync generators are wrapped into an async generator."""

    @pytest.mark.asyncio
    async def test_wraps_list(self, agent):
        # Act
        gen = await agent.ensure_async_generator([1, 2, 3])
        items = [item async for item in gen]

        # Assert
        assert items == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_wraps_tuple(self, agent):
        # Act
        gen = await agent.ensure_async_generator((7, 8, 9))
        items = [item async for item in gen]

        # Assert
        assert items == [7, 8, 9]

    @pytest.mark.asyncio
    async def test_wraps_empty_list(self, agent):
        # Act
        gen = await agent.ensure_async_generator([])
        items = [item async for item in gen]

        # Assert
        assert items == []

    @pytest.mark.asyncio
    async def test_wraps_dict_iterating_over_keys(self, agent):
        # Arrange — dicts are iterable (yields keys)
        data = {"a": 1, "b": 2}

        # Act
        gen = await agent.ensure_async_generator(data)
        items = [item async for item in gen]

        # Assert
        assert items == ["a", "b"]


# ── ensure_async_generator: unsupported types ─────────────────────────────────

class TestEnsureAsyncGeneratorUnsupportedTypes:
    """Non-iterable, non-coroutine, non-async-generator inputs raise TypeError."""

    @pytest.mark.asyncio
    async def test_raises_type_error_for_integer(self, agent):
        # Act / Assert
        with pytest.raises(TypeError, match="unsupported type"):
            await agent.ensure_async_generator(42)

    @pytest.mark.asyncio
    async def test_raises_type_error_for_none(self, agent):
        # Act / Assert
        with pytest.raises(TypeError, match="unsupported type"):
            await agent.ensure_async_generator(None)

    @pytest.mark.asyncio
    async def test_raises_type_error_for_float(self, agent):
        # Act / Assert
        with pytest.raises(TypeError, match="unsupported type"):
            await agent.ensure_async_generator(3.14)

    @pytest.mark.asyncio
    async def test_error_message_includes_actual_type(self, agent):
        # Act
        with pytest.raises(TypeError) as exc_info:
            await agent.ensure_async_generator(99)

        # Assert — error message names the bad type
        assert "int" in str(exc_info.value)