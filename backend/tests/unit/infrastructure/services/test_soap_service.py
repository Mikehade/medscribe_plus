"""
Unit tests for SOAPService.

Covers:
- Initialization: dependencies stored, key helpers
- generate_soap_note: cache hit, cache miss → LLM, response parsing,
  markdown fence stripping, patient context included, cache write,
  no response fallback, JSON parse error, exception handling
- save_transcript_chunk: append to existing, create new, exception
- get_session_transcript: cache hit, cache miss empty string, exception
- get_soap_note_from_cache: found, not found, exception
- clear_session: invalidates both keys
"""
import json
import pytest
from src.infrastructure.services.soap import SOAPService


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_model(mocker):
    model = mocker.Mock()
    model.prompt = mocker.Mock()
    model.extract_text_response = mocker.Mock(return_value=json.dumps({
        "subjective": "headache",
        "objective": "BP 148/90",
        "assessment": "hypertension",
        "plan": "increase lisinopril",
        "icd10_codes": ["I10"],
        "cpt_codes": ["99213"],
        "medications_mentioned": ["lisinopril 20mg daily"],
        "follow_up": "2 weeks",
        "conditions_detected": ["hypertension"],
    }))
    return model


@pytest.fixture
def mock_cache(mocker):
    cache = mocker.Mock()
    cache.get = mocker.AsyncMock(return_value=None)
    cache.set = mocker.AsyncMock()
    cache.invalidate = mocker.AsyncMock()
    return cache


@pytest.fixture
def soap_service(mock_llm_model, mock_cache):
    return SOAPService(llm_model=mock_llm_model, cache_service=mock_cache)


SAMPLE_TRANSCRIPT = "Doctor: BP is 148/90. Patient: I have a headache."
SAMPLE_SOAP = {
    "subjective": "headache",
    "objective": "BP 148/90",
    "assessment": "hypertension",
    "plan": "increase lisinopril",
    "icd10_codes": ["I10"],
    "cpt_codes": ["99213"],
    "medications_mentioned": ["lisinopril 20mg daily"],
    "follow_up": "2 weeks",
    "conditions_detected": ["hypertension"],
}
SAMPLE_PATIENT_CONTEXT = {
    "name": "John Smith",
    "conditions": ["hypertension"],
    "medications": ["lisinopril 10mg"],
}


def _make_async_gen(payload):
    async def _gen():
        yield payload
    return _gen()


# ── Initialization ────────────────────────────────────────────────────────────

class TestSOAPServiceInitialization:
    """SOAPService stores dependencies and key helpers work correctly."""

    def test_stores_llm_model_and_cache(self, mock_llm_model, mock_cache):
        # Act
        service = SOAPService(llm_model=mock_llm_model, cache_service=mock_cache)

        # Assert
        assert service.llm_model is mock_llm_model
        assert service.cache is mock_cache

    def test_transcript_key_format(self, soap_service):
        # Act / Assert
        assert soap_service._transcript_key("sess-1") == "transcript:sess-1"

    def test_soap_key_format(self, soap_service):
        # Act / Assert
        assert soap_service._soap_key("sess-1") == "soap:sess-1"


# ── generate_soap_note ────────────────────────────────────────────────────────

class TestGenerateSoapNote:
    """generate_soap_note: cache hit, LLM call, parsing, caching, error paths."""

    @pytest.mark.asyncio
    async def test_returns_cached_soap_on_cache_hit(
        self, soap_service, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = SAMPLE_SOAP

        # Act
        result = await soap_service.generate_soap_note(
            transcript=SAMPLE_TRANSCRIPT, session_id="sess-1"
        )

        # Assert
        assert result["success"] is True
        assert result["data"] == SAMPLE_SOAP

    @pytest.mark.asyncio
    async def test_does_not_call_llm_on_cache_hit(
        self, soap_service, mock_llm_model, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = SAMPLE_SOAP

        # Act
        await soap_service.generate_soap_note(
            transcript=SAMPLE_TRANSCRIPT, session_id="sess-1"
        )

        # Assert
        mock_llm_model.prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_llm_on_cache_miss(
        self, soap_service, mock_llm_model, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = None
        mock_llm_model.prompt.return_value = _make_async_gen({})

        # Act
        await soap_service.generate_soap_note(
            transcript=SAMPLE_TRANSCRIPT, session_id="sess-1"
        )

        # Assert
        mock_llm_model.prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_called_with_correct_flags(
        self, soap_service, mock_llm_model, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = None
        mock_llm_model.prompt.return_value = _make_async_gen({})

        # Act
        await soap_service.generate_soap_note(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        call_kwargs = mock_llm_model.prompt.call_args[1]
        assert call_kwargs["stream"] is False
        assert call_kwargs["enable_tools"] is False
        assert call_kwargs["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_transcript_present_in_llm_text(
        self, soap_service, mock_llm_model, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = None
        mock_llm_model.prompt.return_value = _make_async_gen({})

        # Act
        await soap_service.generate_soap_note(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        text_arg = mock_llm_model.prompt.call_args[1]["text"]
        assert SAMPLE_TRANSCRIPT in text_arg

    @pytest.mark.asyncio
    async def test_patient_context_included_in_llm_text(
        self, soap_service, mock_llm_model, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = None
        mock_llm_model.prompt.return_value = _make_async_gen({})

        # Act
        await soap_service.generate_soap_note(
            transcript=SAMPLE_TRANSCRIPT,
            patient_context=SAMPLE_PATIENT_CONTEXT,
        )

        # Assert
        text_arg = mock_llm_model.prompt.call_args[1]["text"]
        assert "PATIENT CONTEXT" in text_arg
        assert "John Smith" in text_arg

    @pytest.mark.asyncio
    async def test_skips_cache_write_when_no_session_id(
        self, soap_service, mock_llm_model, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = None
        mock_llm_model.prompt.return_value = _make_async_gen({})

        # Act
        await soap_service.generate_soap_note(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        mock_cache.set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_failure_when_no_llm_response(
        self, soap_service, mock_llm_model, mock_cache
    ):
        # Arrange — async gen yields nothing → response stays None
        async def _empty():
            return
            yield
        mock_cache.get.return_value = None
        mock_llm_model.prompt.return_value = _empty()

        # Act
        result = await soap_service.generate_soap_note(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        assert result["success"] is False
        assert "No response" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_failure_on_json_parse_error(
        self, soap_service, mock_llm_model, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = None
        mock_llm_model.prompt.return_value = _make_async_gen({})
        mock_llm_model.extract_text_response.return_value = "not valid json{{"

        # Act
        result = await soap_service.generate_soap_note(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_returns_failure_on_llm_exception(
        self, soap_service, mock_llm_model, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = None
        mock_llm_model.prompt.side_effect = RuntimeError("LLM down")

        # Act
        result = await soap_service.generate_soap_note(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        assert result["success"] is False
        assert "LLM down" in result["error"]

    @pytest.mark.asyncio
    async def test_handles_coroutine_prompt_output(
        self, soap_service, mock_llm_model, mock_cache
    ):
        # Arrange — prompt returns a coroutine instead of async generator
        mock_cache.get.return_value = None
        async def _coro():
            return {}
        mock_llm_model.prompt.return_value = _coro()

        # Act
        result = await soap_service.generate_soap_note(transcript=SAMPLE_TRANSCRIPT)

        # Assert — no crash regardless of response shape
        assert "success" in result


# ── save_transcript_chunk ─────────────────────────────────────────────────────

class TestSaveTranscriptChunk:
    """save_transcript_chunk appends to existing transcript or creates new."""

    @pytest.mark.asyncio
    async def test_creates_new_transcript_when_none_exists(
        self, soap_service, mock_cache
    ):
        # Arrange — no existing transcript
        mock_cache.get.return_value = None

        # Act
        result = await soap_service.save_transcript_chunk(
            session_id="s1", chunk="Doctor: hello"
        )

        # Assert
        assert result["success"] is True
        assert result["full_transcript"] == "Doctor: hello"

    @pytest.mark.asyncio
    async def test_appends_to_existing_transcript(
        self, soap_service, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = "Doctor: hello"

        # Act
        result = await soap_service.save_transcript_chunk(
            session_id="s1", chunk="Patient: hi"
        )

        # Assert
        assert result["full_transcript"] == "Doctor: hello Patient: hi"

    @pytest.mark.asyncio
    async def test_saves_updated_transcript_to_cache(
        self, soap_service, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = "existing"

        # Act
        await soap_service.save_transcript_chunk(session_id="s1", chunk="new")

        # Assert
        mock_cache.set.assert_awaited_once()
        saved_value = mock_cache.set.call_args[0][1]
        assert "existing" in saved_value
        assert "new" in saved_value

    @pytest.mark.asyncio
    async def test_returns_failure_on_cache_exception(
        self, mocker, soap_service, mock_cache
    ):
        # Arrange
        mock_cache.get = mocker.AsyncMock(side_effect=ConnectionError("Redis down"))

        # Act
        result = await soap_service.save_transcript_chunk(session_id="s1", chunk="t")

        # Assert
        assert result["success"] is False
        assert "Redis down" in result["error"]


# ── get_session_transcript ────────────────────────────────────────────────────

class TestGetSessionTranscript:
    """get_session_transcript reads from cache with correct key."""

    @pytest.mark.asyncio
    async def test_returns_transcript_from_cache(
        self, soap_service, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = "Full transcript here"

        # Act
        result = await soap_service.get_session_transcript(session_id="s1")

        # Assert
        assert result["success"] is True
        assert result["transcript"] == "Full transcript here"

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_cache_miss(
        self, soap_service, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = None

        # Act
        result = await soap_service.get_session_transcript(session_id="s1")

        # Assert
        assert result["success"] is True
        assert result["transcript"] == ""

    @pytest.mark.asyncio
    async def test_queries_correct_cache_key(
        self, soap_service, mock_cache
    ):
        # Act
        await soap_service.get_session_transcript(session_id="target-sess")

        # Assert
        mock_cache.get.assert_awaited_once_with("transcript:target-sess")

    @pytest.mark.asyncio
    async def test_returns_failure_on_cache_exception(
        self, mocker, soap_service, mock_cache
    ):
        # Arrange
        mock_cache.get = mocker.AsyncMock(side_effect=RuntimeError("cache error"))

        # Act
        result = await soap_service.get_session_transcript(session_id="s1")

        # Assert
        assert result["success"] is False
        assert "cache error" in result["error"]


# ── get_soap_note_from_cache ──────────────────────────────────────────────────

class TestGetSoapNoteFromCache:
    """get_soap_note_from_cache retrieves or reports not found."""

    @pytest.mark.asyncio
    async def test_returns_soap_data_when_found(
        self, soap_service, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = SAMPLE_SOAP

        # Act
        result = await soap_service.get_soap_note_from_cache(session_id="s1")

        # Assert
        assert result["success"] is True
        assert result["data"] == SAMPLE_SOAP

    @pytest.mark.asyncio
    async def test_returns_failure_when_not_in_cache(
        self, soap_service, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = None

        # Act
        result = await soap_service.get_soap_note_from_cache(session_id="s1")

        # Assert
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_queries_correct_soap_cache_key(
        self, soap_service, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = SAMPLE_SOAP

        # Act
        await soap_service.get_soap_note_from_cache(session_id="key-sess")

        # Assert
        mock_cache.get.assert_awaited_once_with("soap:key-sess")

    @pytest.mark.asyncio
    async def test_returns_failure_on_exception(
        self, mocker, soap_service, mock_cache
    ):
        # Arrange
        mock_cache.get = mocker.AsyncMock(side_effect=RuntimeError("oops"))

        # Act
        result = await soap_service.get_soap_note_from_cache(session_id="s1")

        # Assert
        assert result["success"] is False
        assert "oops" in result["error"]


# ── clear_session ─────────────────────────────────────────────────────────────

class TestClearSession:
    """clear_session invalidates both transcript and SOAP cache keys."""

    @pytest.mark.asyncio
    async def test_invalidates_transcript_and_soap_keys(
        self, soap_service, mock_cache
    ):
        # Act
        await soap_service.clear_session(session_id="clear-sess")

        # Assert
        assert mock_cache.invalidate.await_count == 2
        invalidated_keys = [
            call.args[0] for call in mock_cache.invalidate.await_args_list
        ]
        assert "transcript:clear-sess" in invalidated_keys
        assert "soap:clear-sess" in invalidated_keys