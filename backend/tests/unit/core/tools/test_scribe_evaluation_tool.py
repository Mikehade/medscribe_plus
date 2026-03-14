"""
Unit tests for ScribeEvaluationTools.

"""
import json
import pytest



# ══════════════════════════════════════════════════════════════════════════════
# ScribeEvaluationTools
# ══════════════════════════════════════════════════════════════════════════════

from src.core.tools.scribe_evaluation import ScribeEvaluationTools


@pytest.fixture
def mock_evaluation_agent(mocker):
    agent = mocker.AsyncMock()
    agent.process_message = mocker.AsyncMock(
        return_value={"success": True, "scores": {"overall_ready": True}}
    )
    return agent


@pytest.fixture
def mock_scribe_cache(mocker):
    cache = mocker.AsyncMock()
    cache.get = mocker.AsyncMock(return_value=None)
    return cache


@pytest.fixture
def scribe_eval_tools(mock_evaluation_agent, mock_scribe_cache):
    return ScribeEvaluationTools(
        evaluation_agent=mock_evaluation_agent,
        cache_service=mock_scribe_cache,
        enabled_tools=["evaluate_consultation"],
    )


class TestScribeEvaluationToolsInitialization:
    """ScribeEvaluationTools stores agent and cache correctly."""

    def test_stores_evaluation_agent(self, mock_evaluation_agent, mock_scribe_cache):
        # Act
        tools = ScribeEvaluationTools(
            evaluation_agent=mock_evaluation_agent,
            cache_service=mock_scribe_cache,
        )

        # Assert
        assert tools.evaluation_agent is mock_evaluation_agent
        assert tools.cache_service is mock_scribe_cache

    def test_stores_extra_kwargs(self, mock_evaluation_agent, mock_scribe_cache):
        # Act
        tools = ScribeEvaluationTools(
            evaluation_agent=mock_evaluation_agent,
            cache_service=mock_scribe_cache,
            session_id="s1",
        )

        # Assert
        assert tools.kwargs["session_id"] == "s1"


class TestScribeEvaluationToolsExecute:
    """execute routes to _evaluate_consultation."""

    @pytest.mark.asyncio
    async def test_executes_evaluate_consultation(
        self, scribe_eval_tools, mock_evaluation_agent, mock_scribe_cache
    ):
        # Arrange
        mock_scribe_cache.get.return_value = {"subjective": "headache", "conditions_detected": []}

        # Act
        result = await scribe_eval_tools.execute(
            "evaluate_consultation",
            {"session_id": "s1", "transcript": "Doctor: hi"},
        )

        # Assert
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_returns_error_for_unknown_tool(self, scribe_eval_tools):
        # Act
        result = await scribe_eval_tools.execute("nonexistent", {})

        # Assert
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_catches_unexpected_exception(
        self, scribe_eval_tools, mock_evaluation_agent, mock_scribe_cache
    ):
        # Arrange
        mock_scribe_cache.get.return_value = {}
        mock_evaluation_agent.process_message.side_effect = RuntimeError("agent crashed")

        # Act
        result = await scribe_eval_tools.execute(
            "evaluate_consultation", {"session_id": "s1"}
        )

        # Assert
        assert result["success"] is False
        assert "agent crashed" in result["error"]


class TestEvaluateConsultation:
    """_evaluate_consultation fetches SOAP from cache and calls EvaluationAgent."""

    @pytest.mark.asyncio
    async def test_calls_evaluation_agent_process_message(
        self, scribe_eval_tools, mock_evaluation_agent, mock_scribe_cache
    ):
        # Arrange
        mock_scribe_cache.get.return_value = {"conditions_detected": ["HTN"]}

        # Act
        await scribe_eval_tools._evaluate_consultation(
            session_id="s1", transcript="Doctor: BP is high"
        )

        # Assert
        mock_evaluation_agent.process_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetches_soap_from_cache_using_session_id(
        self, scribe_eval_tools, mock_scribe_cache
    ):
        # Arrange
        mock_scribe_cache.get.return_value = {}

        # Act
        await scribe_eval_tools._evaluate_consultation(session_id="target-sess", transcript="t")

        # Assert
        mock_scribe_cache.get.assert_awaited_once_with("soap:target-sess")

    @pytest.mark.asyncio
    async def test_passes_transcript_and_session_id_in_context_message(
        self, scribe_eval_tools, mock_evaluation_agent, mock_scribe_cache
    ):
        # Arrange
        mock_scribe_cache.get.return_value = {"conditions_detected": ["diabetes"]}

        # Act
        await scribe_eval_tools._evaluate_consultation(
            session_id="ctx-sess", transcript="Patient: I feel tired"
        )

        # Assert — message arg contains session_id and transcript
        call_kwargs = mock_evaluation_agent.process_message.call_args[1]
        context = json.loads(call_kwargs["message"])
        assert context["session_id"] == "ctx-sess"
        assert context["transcript"] == "Patient: I feel tired"

    @pytest.mark.asyncio
    async def test_passes_conditions_from_soap_in_context(
        self, scribe_eval_tools, mock_evaluation_agent, mock_scribe_cache
    ):
        # Arrange
        mock_scribe_cache.get.return_value = {
            "conditions_detected": ["hypertension", "diabetes"]
        }

        # Act
        await scribe_eval_tools._evaluate_consultation(session_id="s1", transcript="t")

        # Assert
        call_kwargs = mock_evaluation_agent.process_message.call_args[1]
        context = json.loads(call_kwargs["message"])
        assert context["conditions"] == ["hypertension", "diabetes"]

    @pytest.mark.asyncio
    async def test_uses_empty_soap_when_cache_miss(
        self, scribe_eval_tools, mock_evaluation_agent, mock_scribe_cache
    ):
        # Arrange — cache returns None
        mock_scribe_cache.get.return_value = None

        # Act
        await scribe_eval_tools._evaluate_consultation(session_id="s1", transcript="t")

        # Assert — context built with empty soap
        call_kwargs = mock_evaluation_agent.process_message.call_args[1]
        context = json.loads(call_kwargs["message"])
        assert context["conditions"] == []

    @pytest.mark.asyncio
    async def test_returns_agent_result_directly(
        self, scribe_eval_tools, mock_evaluation_agent, mock_scribe_cache
    ):
        # Arrange
        mock_scribe_cache.get.return_value = {}
        expected = {"success": True, "scores": {"overall_ready": True}}
        mock_evaluation_agent.process_message.return_value = expected

        # Act
        result = await scribe_eval_tools._evaluate_consultation(session_id="s1", transcript="t")

        # Assert
        assert result == expected