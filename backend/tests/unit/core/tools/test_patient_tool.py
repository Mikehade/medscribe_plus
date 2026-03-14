"""
Unit tests for PatientTools

"""
import json
import pytest

from src.core.tools.patient import PatientTools
 
 
@pytest.fixture
def mock_patient_service(mocker):
    service = mocker.AsyncMock()
    service.get_patient_history = mocker.AsyncMock()
    service.insert_ehr_note = mocker.AsyncMock()
    service.flag_missing_ehr_fields = mocker.AsyncMock()
    return service
 
 
@pytest.fixture
def patient_tools(mock_patient_service):
    return PatientTools(
        patient_service=mock_patient_service,
        enabled_tools=["get_patient_history", "insert_ehr_note", "flag_missing_ehr_fields"],
    )
 
 
PATIENT_HISTORY = {
    "success": True,
    "patient_id": "P001",
    "name": "John Smith",
    "conditions": ["HTN", "diabetes"],
    "medications": ["lisinopril"],
}
SOAP_NOTE_DICT = {"subjective": "chest pain", "plan": "ECG"}
 
 
class TestPatientToolsInitialization:
    """PatientTools stores dependencies correctly."""
 
    def test_stores_patient_service(self, mock_patient_service):
        # Act
        tools = PatientTools(patient_service=mock_patient_service)
 
        # Assert
        assert tools.patient_service is mock_patient_service
 
    def test_stores_extra_kwargs(self, mock_patient_service):
        # Act
        tools = PatientTools(
            patient_service=mock_patient_service,
            patient_id="P042",
            session_id="s1",
        )
 
        # Assert
        assert tools.kwargs["patient_id"] == "P042"
        assert tools.kwargs["session_id"] == "s1"
 
 
class TestPatientToolsExecute:
    """execute routes correctly and normalises results."""
 
    @pytest.mark.asyncio
    async def test_executes_get_patient_history(self, patient_tools, mock_patient_service):
        # Arrange
        mock_patient_service.get_patient_history.return_value = PATIENT_HISTORY
 
        # Act
        result = await patient_tools.execute("get_patient_history", {"patient_id": "P001"})
 
        # Assert
        assert result["success"] is True
 
    @pytest.mark.asyncio
    async def test_returns_error_for_unknown_tool(self, patient_tools):
        # Act
        result = await patient_tools.execute("nonexistent_tool", {})
 
        # Assert
        assert result["success"] is False
        assert "not found" in result["error"]
 
    @pytest.mark.asyncio
    async def test_unpacks_three_tuple_result(self, patient_tools, mock_patient_service):
        # Arrange
        mock_patient_service.get_patient_history.return_value = (True, PATIENT_HISTORY, "OK")
 
        # Act
        result = await patient_tools.execute("get_patient_history", {"patient_id": "P001"})
 
        # Assert
        assert result["success"] is True
        assert result["data"] == PATIENT_HISTORY
        assert result["message"] == "OK"
 
    @pytest.mark.asyncio
    async def test_catches_unexpected_exception(self, patient_tools, mock_patient_service):
        # Arrange
        mock_patient_service.get_patient_history.side_effect = RuntimeError("service down")
 
        # Act
        result = await patient_tools.execute("get_patient_history", {"patient_id": "P001"})
 
        # Assert
        assert result["success"] is False
        assert "service down" in result["error"]
 
    @pytest.mark.asyncio
    async def test_tool_input_overrides_stored_kwargs(self, mock_patient_service):
        # Arrange — baked patient_id should be overridden by tool_input
        tools = PatientTools(patient_service=mock_patient_service, patient_id="P001")
        mock_patient_service.get_patient_history.return_value = PATIENT_HISTORY
 
        # Act
        await tools.execute("get_patient_history", {"patient_id": "P999"})
 
        # Assert — P999 wins because tool_input overrides stored kwargs
        call_kwargs = mock_patient_service.get_patient_history.call_args[1]
        assert call_kwargs["patient_id"] == "P999"
 
 
class TestGetPatientHistory:
    """_get_patient_history passes patient_id and delegates to service."""
 
    @pytest.mark.asyncio
    async def test_returns_patient_history_on_success(self, patient_tools, mock_patient_service):
        # Arrange
        mock_patient_service.get_patient_history.return_value = PATIENT_HISTORY
 
        # Act
        result = await patient_tools._get_patient_history(patient_id="P001")
 
        # Assert
        assert result == PATIENT_HISTORY
 
    @pytest.mark.asyncio
    async def test_passes_patient_id_to_service(self, patient_tools, mock_patient_service):
        # Arrange
        mock_patient_service.get_patient_history.return_value = PATIENT_HISTORY
 
        # Act
        await patient_tools._get_patient_history(patient_id="P042")
 
        # Assert
        mock_patient_service.get_patient_history.assert_awaited_once_with(patient_id="P042")
 
    @pytest.mark.asyncio
    async def test_defaults_to_p001_when_patient_id_absent(self, patient_tools, mock_patient_service):
        # Arrange
        mock_patient_service.get_patient_history.return_value = PATIENT_HISTORY
 
        # Act
        await patient_tools._get_patient_history()
 
        # Assert
        mock_patient_service.get_patient_history.assert_awaited_once_with(patient_id="P001")
 
    @pytest.mark.asyncio
    async def test_returns_failure_on_service_exception(self, patient_tools, mock_patient_service):
        # Arrange
        mock_patient_service.get_patient_history.side_effect = ConnectionError("EHR down")
 
        # Act
        result = await patient_tools._get_patient_history(patient_id="P001")
 
        # Assert
        assert result["success"] is False
        assert "EHR down" in result["error"]
 
 
class TestInsertEhrNote:
    """_insert_ehr_note passes soap_note, patient_id, session_id to service."""
 
    @pytest.mark.asyncio
    async def test_returns_success_on_insert(self, patient_tools, mock_patient_service):
        # Arrange
        mock_patient_service.insert_ehr_note.return_value = {
            "success": True, "ehr_record_id": "EHR-42"
        }
 
        # Act
        result = await patient_tools._insert_ehr_note(
            soap_note=SOAP_NOTE_DICT, patient_id="P001", session_id="s1"
        )
 
        # Assert
        assert result["success"] is True
        assert result["ehr_record_id"] == "EHR-42"

 
    @pytest.mark.asyncio
    async def test_returns_failure_on_service_exception(self, patient_tools, mock_patient_service):
        # Arrange
        mock_patient_service.insert_ehr_note.side_effect = RuntimeError("insert failed")
 
        # Act
        result = await patient_tools._insert_ehr_note(soap_note=SOAP_NOTE_DICT)
 
        # Assert
        assert result["success"] is False
        assert "insert failed" in result["error"]
 
 
class TestFlagMissingEhrFields:
    """_flag_missing_ehr_fields parses JSON string soap_note and delegates."""
 
    @pytest.mark.asyncio
    async def test_returns_missing_fields_on_success(self, patient_tools, mock_patient_service):
        # Arrange
        mock_patient_service.flag_missing_ehr_fields.return_value = {
            "success": True, "missing_fields": ["assessment"]
        }
 
        # Act
        result = await patient_tools._flag_missing_ehr_fields(
            soap_note=SOAP_NOTE_DICT, patient_id="P001"
        )
 
        # Assert
        assert result["success"] is True
        assert "assessment" in result["missing_fields"]
 
    @pytest.mark.asyncio
    async def test_parses_json_string_soap_note(self, patient_tools, mock_patient_service):
        # Arrange — soap_note arrives as a JSON string
        mock_patient_service.flag_missing_ehr_fields.return_value = {"success": True, "missing_fields": []}
        soap_str = json.dumps(SOAP_NOTE_DICT)
 
        # Act
        await patient_tools._flag_missing_ehr_fields(soap_note=soap_str, patient_id="P001")
 
        # Assert — service received a dict, not the raw string
        call_kwargs = mock_patient_service.flag_missing_ehr_fields.call_args[1]
        assert isinstance(call_kwargs["soap_note"], dict)
        assert call_kwargs["soap_note"]["subjective"] == "chest pain"
 
    @pytest.mark.asyncio
    async def test_wraps_invalid_json_string_as_raw_text(self, patient_tools, mock_patient_service):
        # Arrange
        mock_patient_service.flag_missing_ehr_fields.return_value = {"success": True, "missing_fields": []}
 
        # Act
        await patient_tools._flag_missing_ehr_fields(soap_note="not valid json{{", patient_id="P001")
 
        # Assert — wrapped in subjective field
        call_kwargs = mock_patient_service.flag_missing_ehr_fields.call_args[1]
        assert call_kwargs["soap_note"] == {"subjective": "not valid json{{"}
 
    @pytest.mark.asyncio
    async def test_returns_failure_on_service_exception(self, patient_tools, mock_patient_service):
        # Arrange
        mock_patient_service.flag_missing_ehr_fields.side_effect = ValueError("schema error")
 
        # Act
        result = await patient_tools._flag_missing_ehr_fields(soap_note=SOAP_NOTE_DICT)
 
        # Assert
        assert result["success"] is False
        assert "schema error" in result["error"]
 
 