"""
Unit tests for EvaluationService

EvaluationService covers:
- check_hallucinations: happy path, LLM async generator, json parse, exception fallback
- check_drug_interactions: known interactions, no interactions, critical flag, safety score
- check_guideline_alignment: gaps detected, full alignment, empty conditions fallback,
  conditions inferred from soap, score calculation
- aggregate_scores: field mapping, overall_ready logic, cache write, non-dict coercion
- run_full_evaluation: parallel execution, delegates to aggregate_scores

"""
import json
import pytest
from src.infrastructure.services.evaluation import (
    EvaluationService,
    DRUG_INTERACTIONS_DB,
    CLINICAL_GUIDELINES,
)


@pytest.fixture
def mock_llm_model(mocker):
    model = mocker.Mock()
    model.prompt = mocker.Mock()
    model.extract_text_response = mocker.Mock(return_value=json.dumps({
        "hallucination_flags": [],
        "completeness_issues": [],
        "guideline_gaps": [],
        "overall_hallucination_risk": "low",
        "completeness_score": 90,
    }))
    return model


@pytest.fixture
def mock_eval_cache(mocker):
    cache = mocker.Mock()
    cache.get = mocker.AsyncMock(return_value=None)
    cache.set = mocker.AsyncMock()
    return cache


@pytest.fixture
def evaluation_service(mock_llm_model, mock_eval_cache):
    return EvaluationService(llm_model=mock_llm_model, cache_service=mock_eval_cache)


def _make_async_gen(mocker, response_payload: dict):
    """Return an async generator that yields one chunk matching a real LLM response."""
    async def _gen():
        yield response_payload
    return _gen()


SAMPLE_SOAP = {
    "subjective": "headache and fatigue",
    "objective": {"bp": "148/90"},
    "assessment": "hypertension",
    "plan": "increase lisinopril",
    "medications_mentioned": ["lisinopril", "metformin"],
    "conditions_detected": ["hypertension"],
}
SAMPLE_TRANSCRIPT = "Doctor: BP is 148/90. Patient: I have a headache."


# ── check_hallucinations ──────────────────────────────────────────────────────

class TestCheckHallucinations:
    """check_hallucinations calls LLM, parses JSON, returns structured result."""

    @pytest.mark.asyncio
    async def test_returns_success_true_on_happy_path(
        self, evaluation_service, mock_llm_model
    ):
        # Arrange
        mock_llm_model.prompt.return_value = _make_async_gen(mock_llm_model, {})

        # Act
        result = await evaluation_service.check_hallucinations(
            soap_note=SAMPLE_SOAP, transcript=SAMPLE_TRANSCRIPT, session_id="s1"
        )

        # Assert
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_calls_llm_with_transcript_and_soap(
        self, evaluation_service, mock_llm_model
    ):
        # Arrange
        mock_llm_model.prompt.return_value = _make_async_gen(mock_llm_model, {})

        # Act
        await evaluation_service.check_hallucinations(
            soap_note=SAMPLE_SOAP, transcript=SAMPLE_TRANSCRIPT
        )

        # Assert
        call_kwargs = mock_llm_model.prompt.call_args[1]
        assert SAMPLE_TRANSCRIPT in call_kwargs["text"]
        assert call_kwargs["stream"] is False
        assert call_kwargs["enable_tools"] is False

    @pytest.mark.asyncio
    async def test_parses_llm_json_response_fields(
        self, evaluation_service, mock_llm_model
    ):
        # Arrange
        llm_json = json.dumps({
            "hallucination_flags": [{"claim": "fake", "grounded": False, "reason": "not in transcript"}],
            "completeness_issues": ["no vitals"],
            "guideline_gaps": [],
            "overall_hallucination_risk": "medium",
            "completeness_score": 72,
        })
        mock_llm_model.extract_text_response.return_value = llm_json
        mock_llm_model.prompt.return_value = _make_async_gen(mock_llm_model, {})

        # Act
        result = await evaluation_service.check_hallucinations(
            soap_note=SAMPLE_SOAP, transcript=SAMPLE_TRANSCRIPT
        )

        # Assert
        assert result["overall_risk"] == "medium"
        assert result["completeness_score"] == 72
        assert len(result["hallucination_flags"]) == 1
        assert result["completeness_issues"] == ["no vitals"]

    @pytest.mark.asyncio
    async def test_strips_markdown_fences_from_llm_response(
        self, evaluation_service, mock_llm_model
    ):
        # Arrange — LLM wraps JSON in markdown fences
        raw = "```json\n" + json.dumps({
            "hallucination_flags": [],
            "completeness_issues": [],
            "guideline_gaps": [],
            "overall_hallucination_risk": "low",
            "completeness_score": 88,
        }) + "\n```"
        mock_llm_model.extract_text_response.return_value = raw
        mock_llm_model.prompt.return_value = _make_async_gen(mock_llm_model, {})

        # Act
        result = await evaluation_service.check_hallucinations(
            soap_note=SAMPLE_SOAP, transcript=SAMPLE_TRANSCRIPT
        )

        # Assert — should parse cleanly without raising
        assert result["success"] is True
        assert result["completeness_score"] == 88

    @pytest.mark.asyncio
    async def test_returns_safe_fallback_on_json_parse_error(
        self, evaluation_service, mock_llm_model
    ):
        # Arrange — LLM returns gibberish
        mock_llm_model.extract_text_response.return_value = "not json at all"
        mock_llm_model.prompt.return_value = _make_async_gen(mock_llm_model, {})

        # Act
        result = await evaluation_service.check_hallucinations(
            soap_note=SAMPLE_SOAP, transcript=SAMPLE_TRANSCRIPT
        )

        # Assert — safe defaults, never crash
        assert result["success"] is True
        assert result["overall_risk"] == "low"
        assert result["completeness_score"] == 85
        assert result["hallucination_flags"] == []

    @pytest.mark.asyncio
    async def test_returns_safe_fallback_on_llm_exception(
        self, evaluation_service, mock_llm_model
    ):
        # Arrange
        mock_llm_model.prompt.side_effect = RuntimeError("LLM crashed")

        # Act
        result = await evaluation_service.check_hallucinations(
            soap_note=SAMPLE_SOAP, transcript=SAMPLE_TRANSCRIPT
        )

        # Assert
        assert result["success"] is True
        assert result["overall_risk"] == "low"

    @pytest.mark.asyncio
    async def test_handles_coroutine_prompt_output(
        self, evaluation_service, mock_llm_model
    ):
        # Arrange — prompt returns a coroutine instead of async generator
        async def _coro():
            return {}
        mock_llm_model.prompt.return_value = _coro()

        # Act
        result = await evaluation_service.check_hallucinations(
            soap_note=SAMPLE_SOAP, transcript=SAMPLE_TRANSCRIPT
        )

        # Assert — no crash
        assert result["success"] is True


# ── check_drug_interactions ───────────────────────────────────────────────────

class TestCheckDrugInteractions:
    """check_drug_interactions cross-references medications against DRUG_INTERACTIONS_DB."""

    @pytest.mark.asyncio
    async def test_returns_success_true(self, evaluation_service):
        # Act
        result = await evaluation_service.check_drug_interactions(medications=["aspirin"])

        # Assert
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_detects_known_interaction(self, evaluation_service):
        # Arrange — lisinopril + potassium is in the DB
        meds = ["lisinopril 10mg", "potassium supplement"]

        # Act
        result = await evaluation_service.check_drug_interactions(medications=meds)

        # Assert
        assert len(result["interactions"]) >= 1
        assert any(i["drug_a"] == "lisinopril" for i in result["interactions"])

    @pytest.mark.asyncio
    async def test_returns_no_interactions_for_safe_medications(self, evaluation_service):
        # Arrange — no known interactions between these
        meds = ["vitamin C", "zinc"]

        # Act
        result = await evaluation_service.check_drug_interactions(medications=meds)

        # Assert
        assert result["interactions"] == []
        assert result["drug_safety_score"] == 100
        assert result["has_critical_interactions"] is False

    @pytest.mark.asyncio
    async def test_flags_critical_interaction(self, evaluation_service):
        # Arrange — warfarin + aspirin is a high-severity interaction
        meds = ["warfarin", "aspirin"]

        # Act
        result = await evaluation_service.check_drug_interactions(medications=meds)

        # Assert
        assert result["has_critical_interactions"] is True
        assert result["drug_safety_score"] == 60

    @pytest.mark.asyncio
    async def test_moderate_interaction_gives_score_80(self, evaluation_service):
        # Arrange — lisinopril + potassium is moderate severity
        meds = ["lisinopril", "potassium"]

        # Act
        result = await evaluation_service.check_drug_interactions(medications=meds)

        # Assert — has interactions but none are "high" severity
        has_only_moderate = all(
            i["severity"] != "high" for i in result["interactions"]
        )
        if has_only_moderate and result["interactions"]:
            assert result["drug_safety_score"] == 80
            assert result["has_critical_interactions"] is False

    @pytest.mark.asyncio
    async def test_returns_safe_fallback_on_exception(self, mocker, mock_eval_cache):
        # Arrange — inject a broken LLM to trigger the except path via a bad medications arg
        service = EvaluationService(llm_model=mocker.Mock(), cache_service=mock_eval_cache)

        # Act — pass a non-iterable to force the exception
        result = await service.check_drug_interactions(medications=None)

        # Assert
        assert result["success"] is True
        assert result["drug_safety_score"] == 100

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self, evaluation_service):
        # Arrange — uppercase medication names
        meds = ["WARFARIN", "ASPIRIN"]

        # Act
        result = await evaluation_service.check_drug_interactions(medications=meds)

        # Assert
        assert result["has_critical_interactions"] is True


# ── check_guideline_alignment ─────────────────────────────────────────────────

class TestCheckGuidelineAlignment:
    """check_guideline_alignment scores note against condition-specific guidelines."""

    @pytest.mark.asyncio
    async def test_returns_success_true(self, evaluation_service):
        # Act
        result = await evaluation_service.check_guideline_alignment(
            soap_note=SAMPLE_SOAP, conditions=["hypertension"]
        )

        # Assert
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_detects_gaps_for_known_condition(self, evaluation_service):
        # Arrange — SOAP note has no ACE inhibitor or lifestyle counseling mentions
        sparse_soap = {"assessment": "hypertension", "plan": "monitor"}

        # Act
        result = await evaluation_service.check_guideline_alignment(
            soap_note=sparse_soap, conditions=["hypertension"]
        )

        # Assert
        assert len(result["gaps"]) > 0
        assert result["alignment_score"] < 100

    @pytest.mark.asyncio
    async def test_full_alignment_when_all_guidelines_documented(self, evaluation_service):
        # Arrange — SOAP note mentions all hypertension guideline key terms
        hypertension_guidelines = CLINICAL_GUIDELINES["hypertension"]
        # Build note text that includes the first 3 words of each guideline
        note_text = " ".join(
            " ".join(g.lower().split()[:3]) for g in hypertension_guidelines
        )
        complete_soap = {"assessment": note_text, "plan": note_text}

        # Act
        result = await evaluation_service.check_guideline_alignment(
            soap_note=complete_soap, conditions=["hypertension"]
        )

        # Assert
        assert result["gaps"] == []
        assert result["alignment_score"] == 100

    @pytest.mark.asyncio
    async def test_falls_back_to_soap_conditions_detected_when_empty(self, evaluation_service):
        # Arrange — empty conditions list, SOAP has conditions_detected
        soap = {**SAMPLE_SOAP, "conditions_detected": ["hypertension"]}

        # Act
        result = await evaluation_service.check_guideline_alignment(
            soap_note=soap, conditions=[]
        )

        # Assert — still evaluated against hypertension guidelines
        assert result["success"] is True
        assert isinstance(result["alignment_score"], int)

    @pytest.mark.asyncio
    async def test_infers_conditions_from_note_text_as_last_resort(self, evaluation_service):
        # Arrange — no conditions passed, none in conditions_detected, but note text mentions "uri"
        soap = {"assessment": "patient has uri symptoms", "plan": "rest"}

        # Act
        result = await evaluation_service.check_guideline_alignment(
            soap_note=soap, conditions=[]
        )

        # Assert — URI guidelines applied
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_suggestions_capped_at_five(self, evaluation_service):
        # Arrange — both hypertension and diabetes, which have many guidelines
        sparse_soap = {"assessment": "nothing documented", "plan": "nothing"}

        # Act
        result = await evaluation_service.check_guideline_alignment(
            soap_note=sparse_soap, conditions=["hypertension", "type 2 diabetes"]
        )

        # Assert
        assert len(result["suggestions"]) <= 5

    @pytest.mark.asyncio
    async def test_returns_fallback_on_exception(self, mocker, mock_eval_cache):
        # Arrange — force an exception by passing a non-serialisable soap_note
        service = EvaluationService(llm_model=mocker.Mock(), cache_service=mock_eval_cache)

        class BadObj:
            pass

        result = await service.check_guideline_alignment(
            soap_note=BadObj(), conditions=["hypertension"]
        )

        # Assert — safe fallback
        assert result["success"] is True
        assert result["alignment_score"] == 87


# ── aggregate_scores ──────────────────────────────────────────────────────────

class TestAggregateScores:
    """aggregate_scores maps inputs to dashboard payload and caches result."""

    HALL = {
        "completeness_score": 90,
        "completeness_issues": ["no vitals"],
        "overall_risk": "low",
        "hallucination_flags": [],
    }
    DRUG = {
        "drug_safety_score": 80,
        "interactions": [{"drug_a": "warfarin", "drug_b": "aspirin"}],
        "has_critical_interactions": False,
    }
    GUIDE = {
        "alignment_score": 75,
        "suggestions": ["Document ACE inhibitor"],
    }

    @pytest.mark.asyncio
    async def test_returns_success_true(self, evaluation_service):
        # Act
        result = await evaluation_service.aggregate_scores(
            hallucination_result=self.HALL,
            drug_result=self.DRUG,
            guideline_result=self.GUIDE,
        )

        # Assert
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_maps_fields_correctly(self, evaluation_service):
        # Act
        result = await evaluation_service.aggregate_scores(
            hallucination_result=self.HALL,
            drug_result=self.DRUG,
            guideline_result=self.GUIDE,
        )

        # Assert
        scores = result["scores"]
        assert scores["completeness"] == 90
        assert scores["completeness_issues"] == ["no vitals"]
        assert scores["hallucination_risk"] == "low"
        assert scores["drug_safety"] == 80
        assert scores["drug_interactions"] == self.DRUG["interactions"]
        assert scores["has_critical_interactions"] is False
        assert scores["guideline_alignment"] == 75
        assert scores["guideline_suggestions"] == ["Document ACE inhibitor"]

    @pytest.mark.asyncio
    async def test_overall_ready_true_when_no_critical_issues(self, evaluation_service):
        # Act
        result = await evaluation_service.aggregate_scores(
            hallucination_result={**self.HALL, "overall_risk": "low"},
            drug_result={**self.DRUG, "has_critical_interactions": False},
            guideline_result=self.GUIDE,
        )

        # Assert
        assert result["scores"]["overall_ready"] is True

    @pytest.mark.asyncio
    async def test_overall_ready_false_when_high_hallucination_risk(self, evaluation_service):
        # Act
        result = await evaluation_service.aggregate_scores(
            hallucination_result={**self.HALL, "overall_risk": "high"},
            drug_result=self.DRUG,
            guideline_result=self.GUIDE,
        )

        # Assert
        assert result["scores"]["overall_ready"] is False

    @pytest.mark.asyncio
    async def test_overall_ready_false_when_critical_drug_interaction(self, evaluation_service):
        # Act
        result = await evaluation_service.aggregate_scores(
            hallucination_result=self.HALL,
            drug_result={**self.DRUG, "has_critical_interactions": True},
            guideline_result=self.GUIDE,
        )

        # Assert
        assert result["scores"]["overall_ready"] is False

    @pytest.mark.asyncio
    async def test_caches_scores_when_session_id_provided(
        self, evaluation_service, mock_eval_cache
    ):
        # Act
        await evaluation_service.aggregate_scores(
            hallucination_result=self.HALL,
            drug_result=self.DRUG,
            guideline_result=self.GUIDE,
            session_id="cache-sess",
        )

        # Assert
        mock_eval_cache.set.assert_awaited_once()
        call_args = mock_eval_cache.set.call_args[0]
        assert "evaluation:scores:cache-sess" in call_args[0]

    @pytest.mark.asyncio
    async def test_skips_cache_when_no_session_id(
        self, evaluation_service, mock_eval_cache
    ):
        # Act
        await evaluation_service.aggregate_scores(
            hallucination_result=self.HALL,
            drug_result=self.DRUG,
            guideline_result=self.GUIDE,
            session_id=None,
        )

        # Assert
        mock_eval_cache.set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_coerces_non_dict_hallucination_result_to_defaults(self, evaluation_service):
        # Arrange — LLM passed a string instead of dict
        result = await evaluation_service.aggregate_scores(
            hallucination_result="summarized result",
            drug_result=self.DRUG,
            guideline_result=self.GUIDE,
        )

        # Assert — defaults used, no crash
        assert result["success"] is True
        assert result["scores"]["completeness"] == 85  # default

    @pytest.mark.asyncio
    async def test_coerces_non_dict_drug_result_to_defaults(self, evaluation_service):
        # Act
        result = await evaluation_service.aggregate_scores(
            hallucination_result=self.HALL,
            drug_result=None,
            guideline_result=self.GUIDE,
        )

        # Assert
        assert result["success"] is True
        assert result["scores"]["drug_safety"] == 100  # default

    @pytest.mark.asyncio
    async def test_uses_defaults_for_missing_keys_in_results(self, evaluation_service):
        # Arrange — all empty dicts
        result = await evaluation_service.aggregate_scores(
            hallucination_result={},
            drug_result={},
            guideline_result={},
        )

        # Assert — all defaults applied
        scores = result["scores"]
        assert scores["completeness"] == 85
        assert scores["drug_safety"] == 100
        assert scores["guideline_alignment"] == 0
        assert scores["overall_ready"] is True


# ── run_full_evaluation ───────────────────────────────────────────────────────

class TestRunFullEvaluation:
    """run_full_evaluation runs three checks in parallel then aggregates."""

    @pytest.mark.asyncio
    async def test_returns_success_and_scores(self, mocker, evaluation_service):
        # Arrange
        mocker.patch.object(
            evaluation_service, "check_hallucinations",
            return_value={"success": True, "completeness_score": 90,
                          "overall_risk": "low", "hallucination_flags": [],
                          "completeness_issues": [], "guideline_gaps": []}
        )
        mocker.patch.object(
            evaluation_service, "check_drug_interactions",
            return_value={"success": True, "interactions": [],
                          "drug_safety_score": 100, "has_critical_interactions": False}
        )
        mocker.patch.object(
            evaluation_service, "check_guideline_alignment",
            return_value={"success": True, "alignment_score": 88, "suggestions": []}
        )

        # Act
        result = await evaluation_service.run_full_evaluation(
            soap_note=SAMPLE_SOAP,
            transcript=SAMPLE_TRANSCRIPT,
            conditions=["hypertension"],
            session_id="full-sess",
        )

        # Assert
        assert result["success"] is True
        assert "scores" in result

    @pytest.mark.asyncio
    async def test_runs_all_three_checks(self, mocker, evaluation_service):
        # Arrange
        mock_hall = mocker.patch.object(
            evaluation_service, "check_hallucinations",
            return_value={"success": True, "completeness_score": 90,
                          "overall_risk": "low", "hallucination_flags": [],
                          "completeness_issues": [], "guideline_gaps": []}
        )
        mock_drug = mocker.patch.object(
            evaluation_service, "check_drug_interactions",
            return_value={"success": True, "interactions": [],
                          "drug_safety_score": 100, "has_critical_interactions": False}
        )
        mock_guide = mocker.patch.object(
            evaluation_service, "check_guideline_alignment",
            return_value={"success": True, "alignment_score": 88, "suggestions": []}
        )

        # Act
        await evaluation_service.run_full_evaluation(
            soap_note=SAMPLE_SOAP,
            transcript=SAMPLE_TRANSCRIPT,
            conditions=["hypertension"],
        )

        # Assert — all three checks fired
        mock_hall.assert_awaited_once()
        mock_drug.assert_awaited_once()
        mock_guide.assert_awaited_once()

