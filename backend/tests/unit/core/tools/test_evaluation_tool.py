"""
Unit tests for EvaluationTools

"""
import json
import pytest


from src.core.tools.evaluation import EvaluationTools


@pytest.fixture
def mock_evaluation_service(mocker):
    service = mocker.AsyncMock()
    service.check_hallucinations = mocker.AsyncMock()
    service.check_drug_interactions = mocker.AsyncMock()
    service.check_guideline_alignment = mocker.AsyncMock()
    service.aggregate_scores = mocker.AsyncMock()
    return service


@pytest.fixture
def mock_eval_cache(mocker):
    cache = mocker.AsyncMock()
    cache.get = mocker.AsyncMock(return_value=None)
    cache.set = mocker.AsyncMock()
    return cache


@pytest.fixture
def evaluation_tools(mock_evaluation_service, mock_eval_cache):
    return EvaluationTools(
        evaluation_service=mock_evaluation_service,
        cache_service=mock_eval_cache,
        enabled_tools=[
            "check_hallucinations",
            "check_drug_interactions",
            "check_guideline_alignment",
            "aggregate_scores",
        ],
    )


HALLUCINATION_RESULT = {
    "success": True,
    "completeness": 88,
    "hallucination_risk": "low",
    "hallucination_flags": [],
}
DRUG_RESULT = {
    "success": True,
    "drug_safety": 100,
    "drug_interactions": [],
    "has_critical_interactions": False,
}
GUIDELINE_RESULT = {
    "success": True,
    "guideline_alignment": 90,
    "guideline_suggestions": [],
}
AGGREGATE_RESULT = {
    "success": True,
    "scores": {
        "completeness": 88,
        "drug_safety": 100,
        "guideline_alignment": 90,
        "overall_ready": True,
    },
}
SAMPLE_SOAP_DICT = {
    "subjective": "headache",
    "medications_mentioned": ["lisinopril", "metformin"],
    "conditions_detected": ["hypertension", "diabetes"],
}


class TestEvaluationToolsInitialization:
    """EvaluationTools stores dependencies correctly."""

    def test_stores_evaluation_service(self, mock_evaluation_service, mock_eval_cache):
        # Act
        tools = EvaluationTools(
            evaluation_service=mock_evaluation_service,
            cache_service=mock_eval_cache,
        )

        # Assert
        assert tools.evaluation_service is mock_evaluation_service
        assert tools.cache_service is mock_eval_cache

    def test_stores_extra_kwargs(self, mock_evaluation_service, mock_eval_cache):
        # Act
        tools = EvaluationTools(
            evaluation_service=mock_evaluation_service,
            cache_service=mock_eval_cache,
            session_id="s1",
        )

        # Assert
        assert tools.kwargs["session_id"] == "s1"


class TestEvaluationToolsExecute:
    """execute routes correctly and normalises results."""

    @pytest.mark.asyncio
    async def test_returns_error_for_unknown_tool(self, evaluation_tools):
        # Act
        result = await evaluation_tools.execute("nonexistent", {})

        # Assert
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_catches_unexpected_exception(self, evaluation_tools, mock_eval_cache):
        # Arrange
        mock_eval_cache.get.side_effect = RuntimeError("cache exploded")

        # Act
        result = await evaluation_tools.execute(
            "check_hallucinations", {"session_id": "s1", "transcript": "t"}
        )

        # Assert
        assert result["success"] is False
        assert "cache exploded" in result["error"]

    @pytest.mark.asyncio
    async def test_stores_hallucination_result_in_kwargs(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange
        mock_eval_cache.get.return_value = SAMPLE_SOAP_DICT
        mock_evaluation_service.check_hallucinations.return_value = HALLUCINATION_RESULT

        # Act
        await evaluation_tools.execute(
            "check_hallucinations", {"session_id": "s1", "transcript": "t"}
        )

        # Assert — result stored for aggregate_scores to read later
        assert evaluation_tools.kwargs["_hallucination_result"] == HALLUCINATION_RESULT

    @pytest.mark.asyncio
    async def test_stores_drug_result_in_kwargs(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange
        mock_eval_cache.get.return_value = SAMPLE_SOAP_DICT
        mock_evaluation_service.check_drug_interactions.return_value = DRUG_RESULT

        # Act
        await evaluation_tools.execute("check_drug_interactions", {"session_id": "s1"})

        # Assert
        assert evaluation_tools.kwargs["_drug_result"] == DRUG_RESULT

    @pytest.mark.asyncio
    async def test_stores_guideline_result_in_kwargs(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange
        mock_eval_cache.get.return_value = SAMPLE_SOAP_DICT
        mock_evaluation_service.check_guideline_alignment.return_value = GUIDELINE_RESULT

        # Act
        await evaluation_tools.execute("check_guideline_alignment", {"session_id": "s1"})

        # Assert
        assert evaluation_tools.kwargs["_guideline_result"] == GUIDELINE_RESULT


class TestCheckHallucinations:
    """_check_hallucinations reads SOAP from cache and delegates to service."""

    @pytest.mark.asyncio
    async def test_returns_hallucination_result_on_success(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange
        mock_eval_cache.get.return_value = SAMPLE_SOAP_DICT
        mock_evaluation_service.check_hallucinations.return_value = HALLUCINATION_RESULT

        # Act
        result = await evaluation_tools._check_hallucinations(
            session_id="s1", transcript="Doctor: hi"
        )

        # Assert
        assert result == HALLUCINATION_RESULT

    @pytest.mark.asyncio
    async def test_fetches_soap_from_cache_using_session_id(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange
        mock_eval_cache.get.return_value = SAMPLE_SOAP_DICT
        mock_evaluation_service.check_hallucinations.return_value = HALLUCINATION_RESULT

        # Act
        await evaluation_tools._check_hallucinations(session_id="target-sess", transcript="t")

        # Assert
        mock_eval_cache.get.assert_awaited_once_with("soap:target-sess")

    @pytest.mark.asyncio
    async def test_passes_transcript_to_service(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange
        mock_eval_cache.get.return_value = SAMPLE_SOAP_DICT
        mock_evaluation_service.check_hallucinations.return_value = HALLUCINATION_RESULT

        # Act
        await evaluation_tools._check_hallucinations(
            session_id="s1", transcript="Doctor: noted"
        )

        # Assert
        call_kwargs = mock_evaluation_service.check_hallucinations.call_args[1]
        assert call_kwargs["transcript"] == "Doctor: noted"

    @pytest.mark.asyncio
    async def test_uses_empty_soap_when_cache_miss(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange — cache returns None
        mock_eval_cache.get.return_value = None
        mock_evaluation_service.check_hallucinations.return_value = HALLUCINATION_RESULT

        # Act
        await evaluation_tools._check_hallucinations(session_id="s1", transcript="t")

        # Assert — empty dict used as fallback
        call_kwargs = mock_evaluation_service.check_hallucinations.call_args[1]
        assert call_kwargs["soap_note"] == {}


class TestCheckDrugInteractions:
    """_check_drug_interactions extracts medications from SOAP and delegates."""

    @pytest.mark.asyncio
    async def test_returns_drug_result_on_success(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange
        mock_eval_cache.get.return_value = SAMPLE_SOAP_DICT
        mock_evaluation_service.check_drug_interactions.return_value = DRUG_RESULT

        # Act
        result = await evaluation_tools._check_drug_interactions(session_id="s1")

        # Assert
        assert result == DRUG_RESULT

    @pytest.mark.asyncio
    async def test_passes_medications_from_soap_to_service(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange
        mock_eval_cache.get.return_value = SAMPLE_SOAP_DICT
        mock_evaluation_service.check_drug_interactions.return_value = DRUG_RESULT

        # Act
        await evaluation_tools._check_drug_interactions(session_id="s1")

        # Assert
        call_kwargs = mock_evaluation_service.check_drug_interactions.call_args[1]
        assert call_kwargs["medications"] == ["lisinopril", "metformin"]

    @pytest.mark.asyncio
    async def test_uses_empty_medications_on_cache_miss(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange
        mock_eval_cache.get.return_value = None
        mock_evaluation_service.check_drug_interactions.return_value = DRUG_RESULT

        # Act
        await evaluation_tools._check_drug_interactions(session_id="s1")

        # Assert
        call_kwargs = mock_evaluation_service.check_drug_interactions.call_args[1]
        assert call_kwargs["medications"] == []


class TestCheckGuidelineAlignment:
    """_check_guideline_alignment extracts conditions from SOAP and delegates."""

    @pytest.mark.asyncio
    async def test_returns_guideline_result_on_success(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange
        mock_eval_cache.get.return_value = SAMPLE_SOAP_DICT
        mock_evaluation_service.check_guideline_alignment.return_value = GUIDELINE_RESULT

        # Act
        result = await evaluation_tools._check_guideline_alignment(session_id="s1")

        # Assert
        assert result == GUIDELINE_RESULT

    @pytest.mark.asyncio
    async def test_passes_conditions_from_soap_to_service(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange
        mock_eval_cache.get.return_value = SAMPLE_SOAP_DICT
        mock_evaluation_service.check_guideline_alignment.return_value = GUIDELINE_RESULT

        # Act
        await evaluation_tools._check_guideline_alignment(session_id="s1")

        # Assert
        call_kwargs = mock_evaluation_service.check_guideline_alignment.call_args[1]
        assert call_kwargs["conditions"] == ["hypertension", "diabetes"]

    @pytest.mark.asyncio
    async def test_uses_empty_conditions_on_cache_miss(
        self, evaluation_tools, mock_evaluation_service, mock_eval_cache
    ):
        # Arrange
        mock_eval_cache.get.return_value = None
        mock_evaluation_service.check_guideline_alignment.return_value = GUIDELINE_RESULT

        # Act
        await evaluation_tools._check_guideline_alignment(session_id="s1")

        # Assert
        call_kwargs = mock_evaluation_service.check_guideline_alignment.call_args[1]
        assert call_kwargs["conditions"] == []


class TestAggregateScores:
    """_aggregate_scores reads stored check results from kwargs and delegates."""

    @pytest.mark.asyncio
    async def test_returns_aggregated_scores_on_success(
        self, evaluation_tools, mock_evaluation_service
    ):
        # Arrange — pre-load check results as execute() would
        evaluation_tools.kwargs["_hallucination_result"] = HALLUCINATION_RESULT
        evaluation_tools.kwargs["_drug_result"] = DRUG_RESULT
        evaluation_tools.kwargs["_guideline_result"] = GUIDELINE_RESULT
        mock_evaluation_service.aggregate_scores.return_value = AGGREGATE_RESULT

        # Act
        result = await evaluation_tools._aggregate_scores(session_id="s1")

        # Assert
        assert result == AGGREGATE_RESULT

    @pytest.mark.asyncio
    async def test_passes_empty_dicts_when_no_results_stored(
        self, evaluation_tools, mock_evaluation_service
    ):
        # Arrange — no check results in kwargs
        mock_evaluation_service.aggregate_scores.return_value = AGGREGATE_RESULT

        # Act
        await evaluation_tools._aggregate_scores(session_id="s1")

        # Assert — empty dicts used as safe defaults
        call_kwargs = mock_evaluation_service.aggregate_scores.call_args[1]
        assert call_kwargs["hallucination_result"] == {}
        assert call_kwargs["drug_result"] == {}
        assert call_kwargs["guideline_result"] == {}

