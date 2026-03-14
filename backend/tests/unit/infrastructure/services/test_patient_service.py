"""
Unit tests for PatientService.

PatientService covers:
- get_patient_history: cache hit, cache miss → mock EHR, unknown patient fallback, exception
- insert_ehr_note: record_id format, cache append, exception handling
- flag_missing_ehr_fields: all present, some missing, LLM extraction called with soap text, exception
"""
import json
import pytest
from src.infrastructure.services.patient import PatientService, MOCK_EHR_DB


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_patient_cache(mocker):
    cache = mocker.Mock()
    cache.get = mocker.AsyncMock(return_value=None)
    cache.set = mocker.AsyncMock()
    return cache


@pytest.fixture
def mock_llm_service(mocker):
    service = mocker.AsyncMock()
    service.prompt_llm_for_text = mocker.AsyncMock(return_value={
        "subjective": "headache",
        "objective": "BP 148/90",
        "assessment": "hypertension",
        "plan": "increase lisinopril",
        "follow_up": "two weeks",
    })
    return service


@pytest.fixture
def patient_service(mock_patient_cache, mock_llm_service):
    return PatientService(
        cache_service=mock_patient_cache,
        llm_service=mock_llm_service,
        ehr_fields_extraction_prompt="Extract SOAP fields.",
        ehr_fields_guided_json={"type": "object"},
    )


COMPLETE_SOAP = {
    "subjective": "headache and fatigue",
    "objective": "BP 148/90, HR 82",
    "assessment": "hypertension",
    "plan": "increase lisinopril to 20mg",
    "follow_up": "in two weeks",
}


# ── get_patient_history ───────────────────────────────────────────────────────

class TestGetPatientHistory:
    """get_patient_history: cache hit, cache miss → EHR, unknown patient, exception."""

    @pytest.mark.asyncio
    async def test_returns_cached_data_on_cache_hit(
        self, mocker, mock_llm_service, mock_patient_cache
    ):
        # Arrange
        cached_patient = {"name": "Cached Patient", "conditions": ["HTN"]}
        mock_patient_cache.get = mocker.AsyncMock(return_value=cached_patient)
        service = PatientService(
            cache_service=mock_patient_cache,
            llm_service=mock_llm_service,
            ehr_fields_extraction_prompt="prompt",
            ehr_fields_guided_json={},
        )

        # Act
        result = await service.get_patient_history(patient_id="P001")

        # Assert
        assert result["success"] is True
        assert result["data"] == cached_patient

    @pytest.mark.asyncio
    async def test_returns_mock_ehr_on_cache_miss(
        self, patient_service, mock_patient_cache
    ):
        # Arrange — cache returns None (miss)
        mock_patient_cache.get.return_value = None

        # Act
        result = await patient_service.get_patient_history(patient_id="P001")

        # Assert
        assert result["success"] is True
        assert result["data"]["name"] == MOCK_EHR_DB["P001"]["name"]

    @pytest.mark.asyncio
    async def test_caches_patient_data_after_ehr_lookup(
        self, patient_service, mock_patient_cache
    ):
        # Arrange
        mock_patient_cache.get.return_value = None

        # Act
        await patient_service.get_patient_history(patient_id="P001")

        # Assert
        mock_patient_cache.set.assert_awaited_once()
        cache_key = mock_patient_cache.set.call_args[0][0]
        assert "patient:context:P001" in cache_key

    @pytest.mark.asyncio
    async def test_falls_back_to_p001_for_unknown_patient(
        self, patient_service, mock_patient_cache
    ):
        # Arrange — patient not in mock EHR, falls back to P001
        mock_patient_cache.get.return_value = None

        # Act
        result = await patient_service.get_patient_history(patient_id="P999")

        # Assert — P001 data used as fallback
        assert result["success"] is True
        assert result["data"]["name"] == MOCK_EHR_DB["P001"]["name"]

    @pytest.mark.asyncio
    async def test_returns_failure_on_cache_exception(
        self, mocker, mock_llm_service, mock_patient_cache
    ):
        # Arrange
        mock_patient_cache.get = mocker.AsyncMock(side_effect=ConnectionError("Redis down"))
        service = PatientService(
            cache_service=mock_patient_cache,
            llm_service=mock_llm_service,
            ehr_fields_extraction_prompt="prompt",
            ehr_fields_guided_json={},
        )

        # Act
        result = await service.get_patient_history(patient_id="P001")

        # Assert
        assert result["success"] is False
        assert "Redis down" in result["error"]

    @pytest.mark.asyncio
    async def test_cache_checked_with_correct_key(
        self, patient_service, mock_patient_cache
    ):
        # Act
        await patient_service.get_patient_history(patient_id="P001")

        # Assert
        mock_patient_cache.get.assert_awaited_once_with("patient:context:P001")


# ── insert_ehr_note ───────────────────────────────────────────────────────────

class TestInsertEhrNote:
    """insert_ehr_note appends note to cache list and returns a record ID."""

    @pytest.mark.asyncio
    async def test_returns_success_and_record_id(
        self, patient_service, mock_patient_cache
    ):
        # Arrange
        mock_patient_cache.get.return_value = []  # existing notes list

        # Act
        result = await patient_service.insert_ehr_note(
            patient_id="P001", soap_note=COMPLETE_SOAP, session_id="sess-1"
        )

        # Assert
        assert result["success"] is True
        assert "ehr_record_id" in result
        assert "NOTE-sess-1" in result["ehr_record_id"]

    @pytest.mark.asyncio
    async def test_record_id_contains_session_id(
        self, patient_service, mock_patient_cache
    ):
        # Arrange
        mock_patient_cache.get.return_value = []

        # Act
        result = await patient_service.insert_ehr_note(
            patient_id="P001", soap_note=COMPLETE_SOAP, session_id="my-session"
        )

        # Assert
        assert "my-session" in result["ehr_record_id"]

    @pytest.mark.asyncio
    async def test_appends_note_to_existing_list(
        self, patient_service, mock_patient_cache
    ):
        # Arrange — two existing notes
        existing = [{"record_id": "NOTE-old-1"}, {"record_id": "NOTE-old-2"}]
        mock_patient_cache.get.return_value = existing

        # Act
        await patient_service.insert_ehr_note(
            patient_id="P001", soap_note=COMPLETE_SOAP, session_id="new"
        )

        # Assert — cache.set called with 3 notes
        saved_list = mock_patient_cache.set.call_args[0][1]
        assert len(saved_list) == 3

    @pytest.mark.asyncio
    async def test_saves_soap_note_in_cached_record(
        self, patient_service, mock_patient_cache
    ):
        # Arrange
        mock_patient_cache.get.return_value = []

        # Act
        await patient_service.insert_ehr_note(
            patient_id="P001", soap_note=COMPLETE_SOAP, session_id="s1"
        )

        # Assert
        saved_list = mock_patient_cache.set.call_args[0][1]
        assert saved_list[0]["soap_note"] == COMPLETE_SOAP

    @pytest.mark.asyncio
    async def test_returns_failure_on_cache_exception(
        self, mocker, mock_llm_service, mock_patient_cache
    ):
        # Arrange
        mock_patient_cache.get = mocker.AsyncMock(side_effect=RuntimeError("cache error"))
        service = PatientService(
            cache_service=mock_patient_cache,
            llm_service=mock_llm_service,
            ehr_fields_extraction_prompt="prompt",
            ehr_fields_guided_json={},
        )

        # Act
        result = await service.insert_ehr_note(
            patient_id="P001", soap_note=COMPLETE_SOAP, session_id="s1"
        )

        # Assert
        assert result["success"] is False
        assert "cache error" in result["error"]

    @pytest.mark.asyncio
    async def test_uses_uuid_when_no_session_id(
        self, patient_service, mock_patient_cache
    ):
        # Arrange
        mock_patient_cache.get.return_value = []

        # Act
        result = await patient_service.insert_ehr_note(
            patient_id="P001", soap_note=COMPLETE_SOAP, session_id=None
        )

        # Assert — record ID still generated (UUID-based)
        assert result["success"] is True
        assert result["ehr_record_id"].startswith("NOTE-")


# ── flag_missing_ehr_fields ───────────────────────────────────────────────────

class TestFlagMissingEhrFields:
    """flag_missing_ehr_fields uses LLM extraction then checks required fields."""

    @pytest.mark.asyncio
    async def test_returns_no_missing_fields_when_complete(
        self, patient_service, mock_llm_service
    ):
        # Arrange — LLM returns all required fields
        mock_llm_service.prompt_llm_for_text.return_value = {
            "subjective": "headache",
            "objective": "BP 148/90",
            "assessment": "hypertension",
            "plan": "lisinopril",
            "follow_up": "2 weeks",
        }

        # Act
        result = await patient_service.flag_missing_ehr_fields(
            patient_id="P001", soap_note=COMPLETE_SOAP
        )

        # Assert
        assert result["success"] is True
        assert result["missing_fields"] == []
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_detects_missing_follow_up_field(
        self, patient_service, mock_llm_service
    ):
        # Arrange — LLM returns soap without follow_up
        mock_llm_service.prompt_llm_for_text.return_value = {
            "subjective": "headache",
            "objective": "BP 148/90",
            "assessment": "hypertension",
            "plan": "lisinopril",
            "follow_up": "",  # empty
        }

        # Act
        result = await patient_service.flag_missing_ehr_fields(
            patient_id="P001", soap_note=COMPLETE_SOAP
        )

        # Assert
        assert result["success"] is True
        assert any(f["field"] == "follow_up" for f in result["missing_fields"])
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_calls_llm_service_with_soap_text(
        self, patient_service, mock_llm_service
    ):
        # Act
        await patient_service.flag_missing_ehr_fields(
            patient_id="P001", soap_note=COMPLETE_SOAP
        )

        # Assert
        mock_llm_service.prompt_llm_for_text.assert_awaited_once()
        call_args = mock_llm_service.prompt_llm_for_text.call_args[1]
        assert "SOAP Text:" in call_args["text"]
        assert "subjective" in call_args["text"]  # soap serialized into text

    @pytest.mark.asyncio
    async def test_passes_extraction_prompt_to_llm(
        self, patient_service, mock_llm_service
    ):
        # Act
        await patient_service.flag_missing_ehr_fields(
            patient_id="P001", soap_note=COMPLETE_SOAP
        )

        # Assert
        call_args = mock_llm_service.prompt_llm_for_text.call_args[1]
        assert call_args["prompt"] == "Extract SOAP fields."

    @pytest.mark.asyncio
    async def test_passes_response_schema_to_llm(
        self, patient_service, mock_llm_service
    ):
        # Act
        await patient_service.flag_missing_ehr_fields(
            patient_id="P001", soap_note=COMPLETE_SOAP
        )

        # Assert
        call_args = mock_llm_service.prompt_llm_for_text.call_args[1]
        assert call_args["response_schema"] == {"type": "object"}

    @pytest.mark.asyncio
    async def test_detects_multiple_missing_fields(
        self, patient_service, mock_llm_service
    ):
        # Arrange — LLM returns mostly empty extraction
        mock_llm_service.prompt_llm_for_text.return_value = {
            "subjective": "headache",
            "objective": "",
            "assessment": "",
            "plan": "",
            "follow_up": "",
        }

        # Act
        result = await patient_service.flag_missing_ehr_fields(
            patient_id="P001", soap_note={}
        )

        # Assert
        assert len(result["missing_fields"]) == 4
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_returns_failure_on_llm_exception(
        self, mocker, mock_patient_cache
    ):
        # Arrange
        bad_llm = mocker.AsyncMock()
        bad_llm.prompt_llm_for_text = mocker.AsyncMock(
            side_effect=RuntimeError("LLM timeout")
        )
        service = PatientService(
            cache_service=mock_patient_cache,
            llm_service=bad_llm,
            ehr_fields_extraction_prompt="prompt",
            ehr_fields_guided_json={},
        )

        # Act
        result = await service.flag_missing_ehr_fields(
            patient_id="P001", soap_note=COMPLETE_SOAP
        )

        # Assert
        assert result["success"] is False
        assert "LLM timeout" in result["error"]