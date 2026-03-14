"""
Unit tests for EvaluationAgent.

Covers:
- Initialization: dependency wiring
- process_message: context parsing, tool injection, LLM loop, cache hit/miss, exception handling
"""
import json
import pytest
from src.core.agents.evaluation import EvaluationAgent


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_model(mocker):
    model = mocker.Mock()
    model.tool_registry = None

    async def _fake_prompt(*args, **kwargs):
        yield {"text": "evaluation complete"}

    model.prompt = mocker.Mock(return_value=_fake_prompt())
    return model


@pytest.fixture
def mock_tool_registry(mocker):
    registry = mocker.Mock()
    registry.get_available_tools = mocker.Mock(return_value=["check_hallucinations", "aggregate_scores"])
    registry.tool_classes = [mocker.Mock(kwargs={}), mocker.Mock(kwargs={})]
    return registry


@pytest.fixture
def mock_prompt_template(mocker):
    template = mocker.Mock()
    template.get_system_prompt = mocker.Mock(return_value="You are a clinical evaluator.")
    return template


@pytest.fixture
def mock_cache(mocker):
    cache = mocker.Mock()
    cache.get = mocker.AsyncMock(return_value=None)
    return cache


@pytest.fixture
def valid_scores():
    return {
        "completeness": 90,
        "hallucination_risk": "low",
        "hallucination_flags": [],
        "drug_safety": 100,
        "drug_interactions": [],
        "has_critical_interactions": False,
        "guideline_alignment": 88,
        "guideline_suggestions": [],
        "completeness_issues": [],
        "overall_ready": True,
    }


@pytest.fixture
def evaluation_agent(mock_llm_model, mock_tool_registry, mock_prompt_template, mock_cache):
    return EvaluationAgent(
        llm_model=mock_llm_model,
        tool_registry=mock_tool_registry,
        prompt_template=mock_prompt_template,
        cache_service=mock_cache,
    )


def build_message(soap="{}", transcript="Doctor: hi", conditions=None, session_id=None):
    payload = {
        "soap_note_json": soap,
        "transcript": transcript,
        "conditions": conditions or ["hypertension"],
    }
    if session_id:
        payload["session_id"] = session_id
    return json.dumps(payload)


# ── Initialization ────────────────────────────────────────────────────────────

class TestEvaluationAgentInitialization:
    """Agent wires dependencies correctly on construction."""

    def test_sets_tool_registry_on_llm_model(
        self, mock_llm_model, mock_tool_registry, mock_prompt_template, mock_cache
    ):
        # Act
        EvaluationAgent(
            llm_model=mock_llm_model,
            tool_registry=mock_tool_registry,
            prompt_template=mock_prompt_template,
            cache_service=mock_cache,
        )

        # Assert
        assert mock_llm_model.tool_registry == mock_tool_registry

    def test_queries_available_tools_on_init(
        self, mock_llm_model, mock_tool_registry, mock_prompt_template, mock_cache
    ):
        # Act
        EvaluationAgent(
            llm_model=mock_llm_model,
            tool_registry=mock_tool_registry,
            prompt_template=mock_prompt_template,
            cache_service=mock_cache,
        )

        # Assert
        mock_tool_registry.get_available_tools.assert_called_once()

    def test_handles_none_tool_registry_without_error(
        self, mock_llm_model, mock_prompt_template, mock_cache
    ):
        # Act
        agent = EvaluationAgent(
            llm_model=mock_llm_model,
            tool_registry=None,
            prompt_template=mock_prompt_template,
            cache_service=mock_cache,
        )

        # Assert
        assert agent.tool_registry is None


# ── process_message: message parsing ─────────────────────────────────────────

class TestProcessMessageParsing:
    """process_message correctly parses the JSON context string."""

    @pytest.mark.asyncio
    async def test_parses_json_string_message(self, mocker, evaluation_agent, mock_cache, valid_scores):
        # Arrange
        mock_cache.get = mocker.AsyncMock(return_value=valid_scores)
        message = build_message(session_id="parse-sess")

        # Act
        result = await evaluation_agent.process_message(message=message)

        # Assert
        assert result["success"] is True
        assert result["session_id"] == "parse-sess"

    @pytest.mark.asyncio
    async def test_parses_dict_message_directly(self, mocker, evaluation_agent, mock_cache, valid_scores):
        # Arrange
        mock_cache.get = mocker.AsyncMock(return_value=valid_scores)
        message = {
            "soap_note_json": "{}",
            "transcript": "hi",
            "conditions": [],
            "session_id": "dict-sess",
        }

        # Act
        result = await evaluation_agent.process_message(message=message)

        # Assert
        assert result["success"] is True
        assert result["session_id"] == "dict-sess"

    @pytest.mark.asyncio
    async def test_bad_json_falls_back_to_empty_context(self, mocker, evaluation_agent, mock_cache, valid_scores):
        # Arrange
        mock_cache.get = mocker.AsyncMock(return_value=valid_scores)

        # Act
        result = await evaluation_agent.process_message(message="not{{valid-json")

        # Assert — should still succeed using empty defaults
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_auto_generates_session_id_when_absent(self, mocker, evaluation_agent, mock_cache, valid_scores):
        # Arrange
        mock_cache.get = mocker.AsyncMock(return_value=valid_scores)
        message = json.dumps({"soap_note_json": "{}", "transcript": "t", "conditions": []})

        # Act
        result = await evaluation_agent.process_message(message=message)

        # Assert — UUID4 is 36 chars
        assert "session_id" in result
        assert len(result["session_id"]) == 36


# ── process_message: context injection ───────────────────────────────────────

class TestProcessMessageContextInjection:
    """Tool kwargs are populated with evaluation context before the LLM loop."""

    @pytest.mark.asyncio
    async def test_injects_context_into_all_tool_kwargs(
        self, mocker, evaluation_agent, mock_tool_registry, mock_cache, valid_scores
    ):
        # Arrange
        mock_cache.get = mocker.AsyncMock(return_value=valid_scores)
        soap = '{"subjective": "headache"}'
        transcript = "Doctor: noted"
        conditions = ["diabetes"]
        session_id = "inject-sess"
        message = json.dumps({
            "soap_note_json": soap,
            "transcript": transcript,
            "conditions": conditions,
            "session_id": session_id,
        })

        # Act
        await evaluation_agent.process_message(message=message)

        # Assert
        for tool in mock_tool_registry.tool_classes:
            assert tool.kwargs["soap_note_json"] == soap
            assert tool.kwargs["transcript"] == transcript
            assert tool.kwargs["conditions"] == conditions
            assert tool.kwargs["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_skips_injection_when_no_registry(
        self, mocker, mock_llm_model, mock_prompt_template, mock_cache, valid_scores
    ):
        # Arrange
        mock_cache.get = mocker.AsyncMock(return_value=valid_scores)
        agent = EvaluationAgent(
            llm_model=mock_llm_model,
            tool_registry=None,
            prompt_template=mock_prompt_template,
            cache_service=mock_cache,
        )

        # Act / Assert — no exception raised
        result = await agent.process_message(message=build_message())
        assert result["success"] is True


# ── process_message: LLM loop ────────────────────────────────────────────────

class TestProcessMessageLLMLoop:
    """LLM.prompt is called with the correct parameters."""

    @pytest.mark.asyncio
    async def test_calls_llm_prompt_once(self, mocker, evaluation_agent, mock_llm_model, mock_cache, valid_scores):
        # Arrange
        mock_cache.get = mocker.AsyncMock(return_value=valid_scores)

        # Act
        await evaluation_agent.process_message(message=build_message())

        # Assert
        mock_llm_model.prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_called_with_stream_false_and_tools_enabled(
        self, mocker, evaluation_agent, mock_llm_model, mock_cache, valid_scores
    ):
        # Arrange
        mock_cache.get = mocker.AsyncMock(return_value=valid_scores)

        # Act
        await evaluation_agent.process_message(message=build_message())

        # Assert
        call_kwargs = mock_llm_model.prompt.call_args[1]
        assert call_kwargs["stream"] is False
        assert call_kwargs["enable_tools"] is True

    @pytest.mark.asyncio
    async def test_llm_called_with_correct_system_prompt(
        self, mocker, evaluation_agent, mock_llm_model, mock_cache, valid_scores
    ):
        # Arrange
        mock_cache.get = mocker.AsyncMock(return_value=valid_scores)

        # Act
        await evaluation_agent.process_message(message=build_message())

        # Assert
        call_kwargs = mock_llm_model.prompt.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a clinical evaluator."

    @pytest.mark.asyncio
    async def test_user_message_contains_soap_and_transcript(
        self, mocker, evaluation_agent, mock_llm_model, mock_cache, valid_scores
    ):
        # Arrange
        mock_cache.get = mocker.AsyncMock(return_value=valid_scores)
        soap = '{"plan": "ECG"}'
        transcript = "Patient: chest pain"
        message = json.dumps({
            "soap_note_json": soap,
            "transcript": transcript,
            "conditions": [],
            "session_id": "txt-sess",
        })

        # Act
        await evaluation_agent.process_message(message=message)

        # Assert
        text_arg = mock_llm_model.prompt.call_args[1]["text"]
        assert soap in text_arg
        assert transcript in text_arg


# ── process_message: cache hit ───────────────────────────────────────────────

class TestProcessMessageCacheHit:
    """Returns scores from cache when aggregate_scores has populated it."""

    @pytest.mark.asyncio
    async def test_returns_scores_from_cache(self, mocker, evaluation_agent, mock_cache, valid_scores):
        # Arrange
        mock_cache.get = mocker.AsyncMock(return_value=valid_scores)

        # Act
        result = await evaluation_agent.process_message(message=build_message(session_id="hit-sess"))

        # Assert
        assert result["success"] is True
        assert result["scores"] == valid_scores

    @pytest.mark.asyncio
    async def test_cache_queried_with_correct_key(self, mocker, evaluation_agent, mock_cache, valid_scores):
        # Arrange
        mock_cache.get = mocker.AsyncMock(return_value=valid_scores)
        session_id = "key-sess"

        # Act
        await evaluation_agent.process_message(message=build_message(session_id=session_id))

        # Assert
        mock_cache.get.assert_called_once_with(f"evaluation:scores:{session_id}")


# ── process_message: cache miss fallback ─────────────────────────────────────

class TestProcessMessageCacheMiss:
    """Falls back to hardcoded defaults when cache returns nothing (None)."""

    @pytest.mark.asyncio
    async def test_returns_success_with_fallback_scores(self, evaluation_agent):
        # Arrange — mock_cache.get returns None by default from fixture

        # Act
        result = await evaluation_agent.process_message(message=build_message())

        # Assert
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_fallback_scores_contain_all_required_keys(self, evaluation_agent):
        # Act
        result = await evaluation_agent.process_message(message=build_message())

        # Assert
        scores = result["scores"]
        for key in ("completeness", "hallucination_risk", "drug_safety", "guideline_alignment", "overall_ready"):
            assert key in scores, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_fallback_scores_reflect_safe_passing_state(self, evaluation_agent):
        # Act
        result = await evaluation_agent.process_message(message=build_message())

        # Assert
        scores = result["scores"]
        assert scores["hallucination_risk"] == "low"
        assert scores["drug_safety"] == 100
        assert scores["overall_ready"] is True
        assert isinstance(scores["hallucination_flags"], list)
        assert isinstance(scores["drug_interactions"], list)


# ── process_message: exception handling ──────────────────────────────────────

class TestProcessMessageExceptionHandling:
    """Exceptions inside the loop are caught and returned as failure dicts."""

    @pytest.mark.asyncio
    async def test_returns_failure_dict_on_llm_exception(
        self, mocker, mock_llm_model, mock_tool_registry, mock_prompt_template, mock_cache
    ):
        # Arrange
        mock_llm_model.prompt = mocker.Mock(side_effect=RuntimeError("LLM exploded"))
        agent = EvaluationAgent(
            llm_model=mock_llm_model,
            tool_registry=mock_tool_registry,
            prompt_template=mock_prompt_template,
            cache_service=mock_cache,
        )

        # Act
        result = await agent.process_message(message=build_message(session_id="err-sess"))

        # Assert
        assert result["success"] is False
        assert "LLM exploded" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_failure_dict_on_cache_exception(
        self, mocker, evaluation_agent, mock_cache
    ):
        # Arrange
        mock_cache.get = mocker.AsyncMock(side_effect=ConnectionError("Redis down"))

        # Act
        result = await evaluation_agent.process_message(message=build_message())

        # Assert
        assert result["success"] is False
        assert "Redis down" in result["error"]

    @pytest.mark.asyncio
    async def test_session_id_always_present_in_failure_result(
        self, mocker, mock_llm_model, mock_tool_registry, mock_prompt_template, mock_cache
    ):
        # Arrange
        mock_llm_model.prompt = mocker.Mock(side_effect=ValueError("bad input"))
        agent = EvaluationAgent(
            llm_model=mock_llm_model,
            tool_registry=mock_tool_registry,
            prompt_template=mock_prompt_template,
            cache_service=mock_cache,
        )
        session_id = "must-survive"

        # Act
        result = await agent.process_message(message=build_message(session_id=session_id))

        # Assert
        assert result["session_id"] == session_id