"""
Unit tests for SOAPTools.

Covers:
- Initialization
- execute: happy path, unknown tool, exception handling, tuple result, non-dict result
- _generate_soap_note: happy path, kwargs threading, exception handling
- _get_session_transcript: happy path, session_id threading, exception handling
"""
import pytest
from src.core.tools.soap import SOAPTools


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_soap_service(mocker):
    service = mocker.AsyncMock()
    service.generate_soap_note = mocker.AsyncMock()
    service.get_session_transcript = mocker.AsyncMock()
    return service


@pytest.fixture
def soap_tools(mock_soap_service):
    return SOAPTools(
        soap_service=mock_soap_service,
        enabled_tools=["generate_soap_note", "get_session_transcript"],
    )


SAMPLE_SOAP = {
    "success": True,
    "subjective": "Patient reports headache",
    "objective": {"bp": "120/80"},
    "assessment": "Tension headache",
    "plan": "Ibuprofen 400mg",
}

SAMPLE_TRANSCRIPT = "Doctor: How are you? Patient: I have a headache."


# ── Initialization ────────────────────────────────────────────────────────────

class TestSOAPToolsInitialization:
    """SOAPTools stores dependencies and kwargs correctly."""

    def test_stores_soap_service(self, mock_soap_service):
        # Act
        tools = SOAPTools(soap_service=mock_soap_service)

        # Assert
        assert tools.soap_service is mock_soap_service

    def test_stores_extra_kwargs(self, mock_soap_service):
        # Act
        tools = SOAPTools(soap_service=mock_soap_service, session_id="s1", transcript="t")

        # Assert
        assert tools.kwargs["session_id"] == "s1"
        assert tools.kwargs["transcript"] == "t"

    def test_initializes_without_enabled_tools(self, mock_soap_service):
        # Act / Assert — no error
        tools = SOAPTools(soap_service=mock_soap_service)
        assert tools.soap_service is not None


# ── execute ───────────────────────────────────────────────────────────────────

class TestSOAPToolsExecute:
    """execute routes to the correct method and normalises the result."""

    @pytest.mark.asyncio
    async def test_executes_generate_soap_note(self, soap_tools, mock_soap_service):
        # Arrange
        mock_soap_service.generate_soap_note.return_value = SAMPLE_SOAP

        # Act
        result = await soap_tools.execute(
            "generate_soap_note",
            {"patient_context": "HTN", "transcript": SAMPLE_TRANSCRIPT, "session_id": "s1"},
        )

        # Assert
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_executes_get_session_transcript(self, soap_tools, mock_soap_service):
        # Arrange
        mock_soap_service.get_session_transcript.return_value = {
            "success": True, "transcript": SAMPLE_TRANSCRIPT
        }

        # Act
        result = await soap_tools.execute("get_session_transcript", {"session_id": "s1"})

        # Assert
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_returns_error_for_unknown_tool(self, soap_tools):
        # Act
        result = await soap_tools.execute("nonexistent_tool", {})

        # Assert
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_wraps_non_dict_result_in_success_dict(self, soap_tools, mock_soap_service):
        # Arrange — service returns a plain string
        mock_soap_service.get_session_transcript.return_value = "raw text"

        # Act
        result = await soap_tools.execute("get_session_transcript", {"session_id": "s1"})

        # Assert
        assert result["success"] is True
        assert result["data"] == "raw text"

    @pytest.mark.asyncio
    async def test_unpacks_three_tuple_result(self, soap_tools, mock_soap_service):
        # Arrange — service returns a (success, data, message) tuple
        mock_soap_service.generate_soap_note.return_value = (True, SAMPLE_SOAP, "OK")

        # Act
        result = await soap_tools.execute(
            "generate_soap_note",
            {"patient_context": "diabetes", "session_id": "s1"},
        )

        # Assert
        assert result["success"] is True
        assert result["data"] == SAMPLE_SOAP
        assert result["message"] == "OK"

    @pytest.mark.asyncio
    async def test_catches_unexpected_exception_in_execute(self, soap_tools, mock_soap_service):
        # Arrange
        mock_soap_service.generate_soap_note.side_effect = RuntimeError("boom")

        # Act
        result = await soap_tools.execute(
            "generate_soap_note",
            {"patient_context": "ctx", "session_id": "s1"},
        )

        # Assert
        assert result["success"] is False
        assert "boom" in result["error"]

    @pytest.mark.asyncio
    async def test_merges_stored_kwargs_with_tool_input(self, mock_soap_service):
        # Arrange — session_id baked into the tool instance kwargs
        tools = SOAPTools(
            soap_service=mock_soap_service,
            session_id="baked-session",
            transcript=SAMPLE_TRANSCRIPT,
        )
        mock_soap_service.get_session_transcript.return_value = {
            "success": True, "transcript": SAMPLE_TRANSCRIPT
        }

        # Act
        await tools.execute("get_session_transcript", {})

        # Assert — service received the baked session_id
        mock_soap_service.get_session_transcript.assert_awaited_once_with(
            session_id="baked-session"
        )


# ── _generate_soap_note ───────────────────────────────────────────────────────

class TestGenerateSoapNote:
    """_generate_soap_note delegates to SOAPService with correct arguments."""

    @pytest.mark.asyncio
    async def test_returns_soap_note_on_success(self, soap_tools, mock_soap_service):
        # Arrange
        mock_soap_service.generate_soap_note.return_value = SAMPLE_SOAP

        # Act
        result = await soap_tools._generate_soap_note(
            patient_context="HTN",
            transcript=SAMPLE_TRANSCRIPT,
            session_id="sess-1",
        )

        # Assert
        assert result == SAMPLE_SOAP

    @pytest.mark.asyncio
    async def test_passes_transcript_and_session_id_to_service(self, soap_tools, mock_soap_service):
        # Arrange
        mock_soap_service.generate_soap_note.return_value = SAMPLE_SOAP

        # Act
        await soap_tools._generate_soap_note(
            patient_context="diabetes",
            transcript=SAMPLE_TRANSCRIPT,
            session_id="sess-abc",
        )

        # Assert
        mock_soap_service.generate_soap_note.assert_awaited_once_with(
            transcript=SAMPLE_TRANSCRIPT,
            patient_context="diabetes",
            session_id="sess-abc",
        )

    @pytest.mark.asyncio
    async def test_uses_empty_string_when_transcript_missing(self, soap_tools, mock_soap_service):
        # Arrange
        mock_soap_service.generate_soap_note.return_value = SAMPLE_SOAP

        # Act
        await soap_tools._generate_soap_note(patient_context="ctx")

        # Assert
        call_kwargs = mock_soap_service.generate_soap_note.call_args[1]
        assert call_kwargs["transcript"] == ""

    @pytest.mark.asyncio
    async def test_uses_none_when_session_id_missing(self, soap_tools, mock_soap_service):
        # Arrange
        mock_soap_service.generate_soap_note.return_value = SAMPLE_SOAP

        # Act
        await soap_tools._generate_soap_note(patient_context="ctx", transcript="t")

        # Assert
        call_kwargs = mock_soap_service.generate_soap_note.call_args[1]
        assert call_kwargs["session_id"] is None

    @pytest.mark.asyncio
    async def test_returns_failure_on_service_exception(self, soap_tools, mock_soap_service):
        # Arrange
        mock_soap_service.generate_soap_note.side_effect = ValueError("LLM failed")

        # Act
        result = await soap_tools._generate_soap_note(
            patient_context="ctx",
            transcript="t",
            session_id="s1",
        )

        # Assert
        assert result["success"] is False
        assert "LLM failed" in result["error"]

    @pytest.mark.asyncio
    async def test_reads_transcript_from_kwargs(self, mock_soap_service):
        # Arrange — transcript stored in instance kwargs, not passed directly
        tools = SOAPTools(
            soap_service=mock_soap_service,
            transcript=SAMPLE_TRANSCRIPT,
            session_id="kw-sess",
        )
        mock_soap_service.generate_soap_note.return_value = SAMPLE_SOAP

        # Act
        await tools._generate_soap_note(patient_context="ctx", **tools.kwargs)

        # Assert
        call_kwargs = mock_soap_service.generate_soap_note.call_args[1]
        assert call_kwargs["transcript"] == SAMPLE_TRANSCRIPT
        assert call_kwargs["session_id"] == "kw-sess"


# ── _get_session_transcript ───────────────────────────────────────────────────

class TestGetSessionTranscript:
    """_get_session_transcript delegates to SOAPService with the correct session_id."""

    @pytest.mark.asyncio
    async def test_returns_transcript_on_success(self, soap_tools, mock_soap_service):
        # Arrange
        mock_soap_service.get_session_transcript.return_value = {
            "success": True,
            "transcript": SAMPLE_TRANSCRIPT,
        }

        # Act
        result = await soap_tools._get_session_transcript(session_id="sess-1")

        # Assert
        assert result["success"] is True
        assert result["transcript"] == SAMPLE_TRANSCRIPT

    @pytest.mark.asyncio
    async def test_passes_session_id_to_service(self, soap_tools, mock_soap_service):
        # Arrange
        mock_soap_service.get_session_transcript.return_value = {"success": True}

        # Act
        await soap_tools._get_session_transcript(session_id="target-sess")

        # Assert
        mock_soap_service.get_session_transcript.assert_awaited_once_with(
            session_id="target-sess"
        )

    @pytest.mark.asyncio
    async def test_uses_empty_string_when_session_id_absent(self, soap_tools, mock_soap_service):
        # Arrange
        mock_soap_service.get_session_transcript.return_value = {"success": True}

        # Act
        await soap_tools._get_session_transcript()

        # Assert
        mock_soap_service.get_session_transcript.assert_awaited_once_with(session_id="")

    @pytest.mark.asyncio
    async def test_returns_failure_on_service_exception(self, soap_tools, mock_soap_service):
        # Arrange
        mock_soap_service.get_session_transcript.side_effect = ConnectionError("DB down")

        # Act
        result = await soap_tools._get_session_transcript(session_id="s1")

        # Assert
        assert result["success"] is False
        assert "DB down" in result["error"]

    @pytest.mark.asyncio
    async def test_get_tool_method_returns_callable_for_valid_tool(self, soap_tools):
        # Act
        method = soap_tools.get_tool_method("generate_soap_note")

        # Assert
        assert method is not None
        assert callable(method)

    @pytest.mark.asyncio
    async def test_get_tool_method_returns_none_for_invalid_tool(self, soap_tools):
        # Act
        method = soap_tools.get_tool_method("does_not_exist")

        # Assert
        assert method is None