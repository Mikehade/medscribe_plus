"""
Unit tests for TranscriptionService.

Covers:
- Initialization: dependencies stored, key helper
- transcribe: delegates to sonic, uses default prompt, empty on failure, returns string
- process_real_time_audio: sonic called, fragment appended, socket event sent,
  empty transcript skipped, sonic failure sends error event, exception sends error event
- _append_transcript: creates new, appends to existing, skips when no cache, exception safe
- get_accumulated_transcript: cache hit, cache miss, no cache, exception safe
- clear_realtime_transcript: invalidates cache key, no cache, exception safe
"""
import pytest
from src.infrastructure.services.transcription import (
    TranscriptionService,
    MEDICAL_SYSTEM_PROMPT,
    TRANSCRIPT_KEY_PREFIX,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_sonic_model(mocker):
    model = mocker.AsyncMock()
    model.transcribe_bytes = mocker.AsyncMock(
        return_value={"success": True, "transcript": "Doctor: hello Patient: hi"}
    )
    return model


@pytest.fixture
def mock_cache(mocker):
    cache = mocker.Mock()
    cache.get = mocker.AsyncMock(return_value=None)
    cache.set = mocker.AsyncMock()
    cache.invalidate = mocker.AsyncMock()
    return cache


@pytest.fixture
def transcription_service(mock_sonic_model, mock_cache):
    return TranscriptionService(
        sonic_model=mock_sonic_model,
        cache_service=mock_cache,
    )


@pytest.fixture
def transcription_service_no_cache(mock_sonic_model):
    return TranscriptionService(sonic_model=mock_sonic_model, cache_service=None)


# ── Initialization ────────────────────────────────────────────────────────────

class TestTranscriptionServiceInitialization:
    """Dependencies stored and key helper works correctly."""

    def test_stores_sonic_model_and_cache(self, mock_sonic_model, mock_cache):
        # Act
        service = TranscriptionService(
            sonic_model=mock_sonic_model, cache_service=mock_cache
        )

        # Assert
        assert service.sonic_model is mock_sonic_model
        assert service.cache is mock_cache

    def test_accepts_none_cache(self, mock_sonic_model):
        # Act
        service = TranscriptionService(sonic_model=mock_sonic_model, cache_service=None)

        # Assert
        assert service.cache is None

    def test_realtime_key_format(self, transcription_service):
        # Act / Assert
        assert transcription_service._realtime_key("P001") == f"{TRANSCRIPT_KEY_PREFIX}:P001"


# ── transcribe ────────────────────────────────────────────────────────────────

class TestTranscribe:
    """transcribe delegates to sonic and returns the transcript string."""

    @pytest.mark.asyncio
    async def test_returns_transcript_string_on_success(
        self, transcription_service, mock_sonic_model
    ):
        # Arrange
        mock_sonic_model.transcribe_bytes.return_value = {
            "success": True,
            "transcript": "Doctor: hello Patient: hi",
        }

        # Act
        result = await transcription_service.transcribe(audio_bytes=b"audio")

        # Assert
        assert result == "Doctor: hello Patient: hi"

    @pytest.mark.asyncio
    async def test_calls_sonic_with_audio_bytes(
        self, transcription_service, mock_sonic_model
    ):
        # Act
        await transcription_service.transcribe(audio_bytes=b"raw audio")

        # Assert
        mock_sonic_model.transcribe_bytes.assert_awaited_once()
        call_kwargs = mock_sonic_model.transcribe_bytes.call_args[1]
        assert call_kwargs["audio_bytes"] == b"raw audio"

    @pytest.mark.asyncio
    async def test_uses_default_medical_system_prompt(
        self, transcription_service, mock_sonic_model
    ):
        # Act
        await transcription_service.transcribe(audio_bytes=b"audio")

        # Assert
        call_kwargs = mock_sonic_model.transcribe_bytes.call_args[1]
        assert call_kwargs["system_prompt"] == MEDICAL_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_uses_custom_system_prompt_when_provided(
        self, transcription_service, mock_sonic_model
    ):
        # Act
        await transcription_service.transcribe(
            audio_bytes=b"audio", system_prompt="Custom prompt"
        )

        # Assert
        call_kwargs = mock_sonic_model.transcribe_bytes.call_args[1]
        assert call_kwargs["system_prompt"] == "Custom prompt"

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_sonic_failure(
        self, transcription_service, mock_sonic_model
    ):
        # Arrange
        mock_sonic_model.transcribe_bytes.return_value = {
            "success": False,
            "error": "Sonic timed out",
        }

        # Act
        result = await transcription_service.transcribe(audio_bytes=b"audio")

        # Assert
        assert result == ""


# ── process_real_time_audio ───────────────────────────────────────────────────

class TestProcessRealTimeAudio:
    """process_real_time_audio transcribes, appends to cache, and sends socket event."""

    @pytest.mark.asyncio
    async def test_calls_sonic_transcribe_bytes(
        self, transcription_service, mock_sonic_model, mocker
    ):
        # Arrange
        send_fn = mocker.AsyncMock()

        # Act
        await transcription_service.process_real_time_audio(
            audio_bytes=b"chunk", patient_id="P001", send_to_socket=send_fn
        )

        # Assert
        mock_sonic_model.transcribe_bytes.assert_awaited_once()
        call_kwargs = mock_sonic_model.transcribe_bytes.call_args[1]
        assert call_kwargs["audio_bytes"] == b"chunk"

    @pytest.mark.asyncio
    async def test_sends_transcript_chunk_event_to_socket(
        self, transcription_service, mock_sonic_model, mocker
    ):
        # Arrange
        mock_sonic_model.transcribe_bytes.return_value = {
            "success": True, "transcript": "Doctor: hello"
        }
        send_fn = mocker.AsyncMock()

        # Act
        await transcription_service.process_real_time_audio(
            audio_bytes=b"chunk", patient_id="P001", send_to_socket=send_fn
        )

        # Assert
        send_fn.assert_awaited_once()
        event = send_fn.call_args[0][0]
        assert event["event"] == "scribe.transcript_chunk"
        assert event["chunk"] == "Doctor: hello"
        assert event["final"] is False

    @pytest.mark.asyncio
    async def test_appends_fragment_to_cache(
        self, transcription_service, mock_sonic_model, mock_cache, mocker
    ):
        # Arrange
        mock_sonic_model.transcribe_bytes.return_value = {
            "success": True, "transcript": "Patient: I feel tired"
        }
        mock_cache.get.return_value = None
        send_fn = mocker.AsyncMock()

        # Act
        await transcription_service.process_real_time_audio(
            audio_bytes=b"chunk", patient_id="P001", send_to_socket=send_fn
        )

        # Assert
        mock_cache.set.assert_awaited_once()
        saved = mock_cache.set.call_args[0][1]
        assert "Patient: I feel tired" in saved

    @pytest.mark.asyncio
    async def test_skips_socket_and_cache_when_transcript_empty(
        self, transcription_service, mock_sonic_model, mock_cache, mocker
    ):
        # Arrange — sonic returns empty transcript (silence)
        mock_sonic_model.transcribe_bytes.return_value = {
            "success": True, "transcript": ""
        }
        send_fn = mocker.AsyncMock()

        # Act
        await transcription_service.process_real_time_audio(
            audio_bytes=b"chunk", patient_id="P001", send_to_socket=send_fn
        )

        # Assert — no socket event, no cache write
        send_fn.assert_not_awaited()
        mock_cache.set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sends_error_event_on_exception(
        self, transcription_service, mock_sonic_model, mocker
    ):
        # Arrange
        mock_sonic_model.transcribe_bytes.side_effect = RuntimeError("crash")
        send_fn = mocker.AsyncMock()

        # Act
        await transcription_service.process_real_time_audio(
            audio_bytes=b"chunk", patient_id="P001", send_to_socket=send_fn
        )

        # Assert
        event = send_fn.call_args[0][0]
        assert event["event"] == "scribe.error"
        assert "crash" in event["error"]


# ── _append_transcript ────────────────────────────────────────────────────────

class TestAppendTranscript:
    """_append_transcript creates or appends the transcript in cache."""

    @pytest.mark.asyncio
    async def test_creates_new_entry_when_cache_empty(
        self, transcription_service, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = None

        # Act
        await transcription_service._append_transcript("key:P001", "Doctor: hello")

        # Assert
        saved = mock_cache.set.call_args[0][1]
        assert saved == "Doctor: hello"

    @pytest.mark.asyncio
    async def test_appends_to_existing_transcript(
        self, transcription_service, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = "Doctor: hello"

        # Act
        await transcription_service._append_transcript("key:P001", "Patient: hi")

        # Assert
        saved = mock_cache.set.call_args[0][1]
        assert saved == "Doctor: hello Patient: hi"

    @pytest.mark.asyncio
    async def test_does_nothing_when_no_cache(self, transcription_service_no_cache):
        # Act / Assert — no error when cache is None
        await transcription_service_no_cache._append_transcript("key", "fragment")

    @pytest.mark.asyncio
    async def test_handles_cache_exception_silently(
        self, mocker, transcription_service, mock_cache
    ):
        # Arrange
        mock_cache.get = mocker.AsyncMock(side_effect=RuntimeError("Redis down"))

        # Act / Assert — must not raise
        await transcription_service._append_transcript("key", "fragment")


# ── get_accumulated_transcript ────────────────────────────────────────────────

class TestGetAccumulatedTranscript:
    """get_accumulated_transcript reads the full real-time transcript from cache."""

    @pytest.mark.asyncio
    async def test_returns_transcript_from_cache(
        self, transcription_service, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = "Full real-time transcript"

        # Act
        result = await transcription_service.get_accumulated_transcript("P001")

        # Assert
        assert result == "Full real-time transcript"

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_cache_miss(
        self, transcription_service, mock_cache
    ):
        # Arrange
        mock_cache.get.return_value = None

        # Act
        result = await transcription_service.get_accumulated_transcript("P001")

        # Assert
        assert result == ""

    @pytest.mark.asyncio
    async def test_queries_correct_cache_key(
        self, transcription_service, mock_cache
    ):
        # Act
        await transcription_service.get_accumulated_transcript("P042")

        # Assert
        mock_cache.get.assert_awaited_once_with(f"{TRANSCRIPT_KEY_PREFIX}:P042")

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_no_cache(
        self, transcription_service_no_cache
    ):
        # Act
        result = await transcription_service_no_cache.get_accumulated_transcript("P001")

        # Assert
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_exception(
        self, mocker, transcription_service, mock_cache
    ):
        # Arrange
        mock_cache.get = mocker.AsyncMock(side_effect=RuntimeError("cache error"))

        # Act
        result = await transcription_service.get_accumulated_transcript("P001")

        # Assert — never raises
        assert result == ""


# ── clear_realtime_transcript ─────────────────────────────────────────────────

class TestClearRealtimeTranscript:
    """clear_realtime_transcript invalidates the patient's transcript cache key."""

    @pytest.mark.asyncio
    async def test_invalidates_correct_cache_key(
        self, transcription_service, mock_cache
    ):
        # Act
        await transcription_service.clear_realtime_transcript("P001")

        # Assert
        mock_cache.invalidate.assert_awaited_once_with(f"{TRANSCRIPT_KEY_PREFIX}:P001")

    @pytest.mark.asyncio
    async def test_does_nothing_when_no_cache(self, transcription_service_no_cache):
        # Act / Assert — no error
        await transcription_service_no_cache.clear_realtime_transcript("P001")

    @pytest.mark.asyncio
    async def test_handles_exception_silently(
        self, mocker, transcription_service, mock_cache
    ):
        # Arrange
        mock_cache.invalidate = mocker.AsyncMock(side_effect=RuntimeError("Redis down"))

        # Act / Assert — must not raise
        await transcription_service.clear_realtime_transcript("P001")