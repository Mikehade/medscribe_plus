"""
Unit tests for SonicModel.

Covers:
- Initialization: credential env vars, attributes
- _initialize_client: creates BedrockRuntimeClient with correct config
- _open_stream: creates fresh client and calls bidirectional stream API
- _send_event: encodes JSON and calls stream.input_stream.send
- _send_session_start / _send_prompt_start / _send_audio_content_start: payload shapes
- _send_audio_chunk: base64 encodes audio correctly
- _send_session_end: sends contentEnd + promptEnd + sessionEnd + closes stream
- _collect_transcript: accumulates USER role text, ignores ASSISTANT + audio
- _decode_to_pcm16: delegates to pydub, sets correct rate/channels/width
- transcribe_bytes: happy path, decode error, stream open error, default prompt
- transcribe_stream: happy path, empty buffer, decode failure, chunk accumulation
"""
import asyncio
import base64
import json
import os
import pytest
from src.infrastructure.language_models.sonic import SonicModel, INPUT_SAMPLE_RATE, CHUNK_SIZE


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sonic():
    return SonicModel(
        aws_access_key="test-key",
        aws_secret_key="test-secret",
        region_name="us-east-1",
        model_id="amazon.nova-2-sonic-v1:0",
    )


@pytest.fixture
def mock_stream(mocker):
    """Async mock representing an open Sonic bidirectional stream."""
    stream = mocker.AsyncMock()
    stream.input_stream = mocker.AsyncMock()
    stream.input_stream.send = mocker.AsyncMock()
    stream.input_stream.close = mocker.AsyncMock()
    return stream


def _response_with_event(mocker, event_dict: dict):
    """Build the nested receive() return value for a given event dict."""
    raw = json.dumps({"event": event_dict}).encode("utf-8")
    value = mocker.Mock()
    value.bytes_ = raw
    result = mocker.Mock()
    result.value = value
    inner = mocker.AsyncMock()
    inner.receive = mocker.AsyncMock(return_value=result)
    return inner


def _make_mock_audio_segment(mocker):
    """
    Build a pydub AudioSegment mock that chains set_* calls correctly.
    _decode_to_pcm16 does:
        audio.set_frame_rate(...).set_channels(...).set_sample_width(...)
    so each method must return the same mock.
    """
    mock_audio = mocker.Mock()
    mock_audio.set_frame_rate = mocker.Mock(return_value=mock_audio)
    mock_audio.set_channels = mocker.Mock(return_value=mock_audio)
    mock_audio.set_sample_width = mocker.Mock(return_value=mock_audio)
    mock_audio.raw_data = b"\x00\x01" * 100
    mock_audio.__len__ = mocker.Mock(return_value=1000)
    return mock_audio


# ── Initialization ────────────────────────────────────────────────────────────

class TestSonicModelInitialization:
    """Credentials are stored and env vars are set on construction."""

    def test_stores_credentials_and_config(self, sonic):
        # Assert
        assert sonic.aws_access_key == "test-key"
        assert sonic.aws_secret_key == "test-secret"
        assert sonic.region_name == "us-east-1"
        assert sonic.model_id == "amazon.nova-2-sonic-v1:0"

    def test_sets_aws_env_vars(self):
        # Act
        SonicModel(aws_access_key="k", aws_secret_key="s", region_name="eu-west-1")

        # Assert
        assert os.environ["AWS_ACCESS_KEY_ID"] == "k"
        assert os.environ["AWS_SECRET_ACCESS_KEY"] == "s"
        assert os.environ["AWS_DEFAULT_REGION"] == "eu-west-1"


# ── _initialize_client ────────────────────────────────────────────────────────

class TestInitializeClient:
    """A fresh BedrockRuntimeClient is created with the correct config."""

    def test_creates_bedrock_runtime_client(self, mocker, sonic):
        # Arrange
        mock_client_cls = mocker.patch("src.infrastructure.language_models.sonic.BedrockRuntimeClient")
        mocker.patch("src.infrastructure.language_models.sonic.Config")
        mocker.patch("src.infrastructure.language_models.sonic.EnvironmentCredentialsResolver")

        # Act
        sonic._initialize_client()

        # Assert
        mock_client_cls.assert_called_once()

    def test_config_uses_correct_endpoint_and_region(self, mocker, sonic):
        # Arrange
        mock_config_cls = mocker.patch("src.infrastructure.language_models.sonic.Config")
        mocker.patch("src.infrastructure.language_models.sonic.BedrockRuntimeClient")
        mocker.patch("src.infrastructure.language_models.sonic.EnvironmentCredentialsResolver")

        # Act
        sonic._initialize_client()

        # Assert
        call_kwargs = mock_config_cls.call_args[1]
        assert "us-east-1" in call_kwargs["endpoint_uri"]
        assert call_kwargs["region"] == "us-east-1"

    def test_returns_new_client_on_each_call(self, mocker, sonic):
        # Arrange — side_effect returns two distinct mocks
        mock_client_cls = mocker.patch("src.infrastructure.language_models.sonic.BedrockRuntimeClient")
        mocker.patch("src.infrastructure.language_models.sonic.Config")
        mocker.patch("src.infrastructure.language_models.sonic.EnvironmentCredentialsResolver")
        mock_client_cls.side_effect = [mocker.Mock(), mocker.Mock()]

        # Act
        client1 = sonic._initialize_client()
        client2 = sonic._initialize_client()

        # Assert — two separate instances, never cached
        assert client1 is not client2
        assert mock_client_cls.call_count == 2


# ── _open_stream ──────────────────────────────────────────────────────────────

class TestOpenStream:
    """_open_stream creates a fresh client and opens a bidirectional stream."""

    @pytest.mark.asyncio
    async def test_calls_invoke_bidirectional_stream(self, mocker, sonic):
        # Arrange
        mock_client = mocker.AsyncMock()
        mock_client.invoke_model_with_bidirectional_stream = mocker.AsyncMock(return_value=mocker.Mock())
        mocker.patch.object(sonic, "_initialize_client", return_value=mock_client)

        # Act
        await sonic._open_stream()

        # Assert
        mock_client.invoke_model_with_bidirectional_stream.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_passes_model_id_to_stream_operation(self, mocker, sonic):
        # Arrange
        mock_client = mocker.AsyncMock()
        mock_client.invoke_model_with_bidirectional_stream = mocker.AsyncMock(return_value=mocker.Mock())
        mocker.patch.object(sonic, "_initialize_client", return_value=mock_client)
        mock_op_cls = mocker.patch(
            "src.infrastructure.language_models.sonic.InvokeModelWithBidirectionalStreamOperationInput"
        )

        # Act
        await sonic._open_stream()

        # Assert
        mock_op_cls.assert_called_once_with(model_id=sonic.model_id)


# ── _send_event ───────────────────────────────────────────────────────────────

class TestSendEvent:
    """_send_event encodes JSON and calls stream.input_stream.send."""

    @pytest.mark.asyncio
    async def test_sends_encoded_event_to_stream(self, mocker, sonic, mock_stream):
        # Arrange
        mock_chunk_cls = mocker.patch(
            "src.infrastructure.language_models.sonic.InvokeModelWithBidirectionalStreamInputChunk"
        )
        mocker.patch("src.infrastructure.language_models.sonic.BidirectionalInputPayloadPart")
        event_json = json.dumps({"event": {"sessionEnd": {}}})

        # Act
        await sonic._send_event(mock_stream, event_json)

        # Assert
        mock_stream.input_stream.send.assert_awaited_once()
        mock_chunk_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_payload_bytes_are_utf8_encoded_json(self, mocker, sonic, mock_stream):
        # Arrange
        mock_part_cls = mocker.patch(
            "src.infrastructure.language_models.sonic.BidirectionalInputPayloadPart"
        )
        mocker.patch("src.infrastructure.language_models.sonic.InvokeModelWithBidirectionalStreamInputChunk")
        event_json = '{"event": {"sessionEnd": {}}}'

        # Act
        await sonic._send_event(mock_stream, event_json)

        # Assert
        call_kwargs = mock_part_cls.call_args[1]
        assert call_kwargs["bytes_"] == event_json.encode("utf-8")


# ── _send_session_start ───────────────────────────────────────────────────────

class TestSendSessionStart:
    """_send_session_start sends a sessionStart event with inference config."""

    @pytest.mark.asyncio
    async def test_sends_exactly_one_event(self, mocker, sonic, mock_stream):
        # Arrange
        sent_events = []

        async def capture(stream, event_json):
            sent_events.append(json.loads(event_json))

        mocker.patch.object(sonic, "_send_event", side_effect=capture)

        # Act
        await sonic._send_session_start(mock_stream)

        # Assert
        assert len(sent_events) == 1
        assert "sessionStart" in sent_events[0]["event"]

    @pytest.mark.asyncio
    async def test_session_start_includes_inference_config(self, mocker, sonic, mock_stream):
        # Arrange
        sent_events = []

        async def capture(stream, event_json):
            sent_events.append(json.loads(event_json))

        mocker.patch.object(sonic, "_send_event", side_effect=capture)

        # Act
        await sonic._send_session_start(mock_stream)

        # Assert
        config = sent_events[0]["event"]["sessionStart"]["inferenceConfiguration"]
        assert "maxTokens" in config
        assert "temperature" in config
        assert "topP" in config


# ── _send_prompt_start ────────────────────────────────────────────────────────

class TestSendPromptStart:
    """_send_prompt_start sends promptStart + system prompt content block (4 events)."""

    @pytest.mark.asyncio
    async def test_sends_four_events(self, mocker, sonic, mock_stream):
        # Arrange
        sent_events = []

        async def capture(stream, event_json):
            sent_events.append(json.loads(event_json))

        mocker.patch.object(sonic, "_send_event", side_effect=capture)

        # Act
        await sonic._send_prompt_start(mock_stream, "p-name", "c-name", "System instruction")

        # Assert — promptStart + contentStart + textInput + contentEnd
        assert len(sent_events) == 4

    @pytest.mark.asyncio
    async def test_first_event_is_prompt_start(self, mocker, sonic, mock_stream):
        # Arrange
        sent_events = []

        async def capture(stream, event_json):
            sent_events.append(json.loads(event_json))

        mocker.patch.object(sonic, "_send_event", side_effect=capture)

        # Act
        await sonic._send_prompt_start(mock_stream, "p-name", "c-name", "sys")

        # Assert
        assert "promptStart" in sent_events[0]["event"]

    @pytest.mark.asyncio
    async def test_system_prompt_text_in_third_event(self, mocker, sonic, mock_stream):
        # Arrange
        sent_events = []

        async def capture(stream, event_json):
            sent_events.append(json.loads(event_json))

        mocker.patch.object(sonic, "_send_event", side_effect=capture)
        system_prompt = "Transcribe accurately."

        # Act
        await sonic._send_prompt_start(mock_stream, "p-name", "c-name", system_prompt)

        # Assert — textInput is the third event (index 2)
        text_input_event = sent_events[2]["event"]["textInput"]
        assert text_input_event["content"] == system_prompt


# ── _send_audio_content_start ─────────────────────────────────────────────────

class TestSendAudioContentStart:
    """_send_audio_content_start opens an AUDIO content block for USER role."""

    @pytest.mark.asyncio
    async def test_sends_audio_type_user_role_event(self, mocker, sonic, mock_stream):
        # Arrange
        sent_events = []

        async def capture(stream, event_json):
            sent_events.append(json.loads(event_json))

        mocker.patch.object(sonic, "_send_event", side_effect=capture)

        # Act
        await sonic._send_audio_content_start(mock_stream, "p-name", "a-name")

        # Assert
        content_start = sent_events[0]["event"]["contentStart"]
        assert content_start["type"] == "AUDIO"
        assert content_start["role"] == "USER"

    @pytest.mark.asyncio
    async def test_audio_config_matches_sonic_requirements(self, mocker, sonic, mock_stream):
        # Arrange
        sent_events = []

        async def capture(stream, event_json):
            sent_events.append(json.loads(event_json))

        mocker.patch.object(sonic, "_send_event", side_effect=capture)

        # Act
        await sonic._send_audio_content_start(mock_stream, "p-name", "a-name")

        # Assert
        audio_cfg = sent_events[0]["event"]["contentStart"]["audioInputConfiguration"]
        assert audio_cfg["sampleRateHertz"] == INPUT_SAMPLE_RATE
        assert audio_cfg["channelCount"] == 1
        assert audio_cfg["encoding"] == "base64"


# ── _send_audio_chunk ─────────────────────────────────────────────────────────

class TestSendAudioChunk:
    """_send_audio_chunk base64-encodes audio and sends an audioInput event."""

    @pytest.mark.asyncio
    async def test_content_is_base64_encoded_audio(self, mocker, sonic, mock_stream):
        # Arrange
        sent_events = []

        async def capture(stream, event_json):
            sent_events.append(json.loads(event_json))

        mocker.patch.object(sonic, "_send_event", side_effect=capture)
        raw_audio = b"\x00\x01\x02\x03"

        # Act
        await sonic._send_audio_chunk(mock_stream, "p-name", "a-name", raw_audio)

        # Assert — round-trip the base64 to verify it matches original bytes
        encoded_content = sent_events[0]["event"]["audioInput"]["content"]
        assert base64.b64decode(encoded_content) == raw_audio

    @pytest.mark.asyncio
    async def test_references_correct_prompt_and_content_names(self, mocker, sonic, mock_stream):
        # Arrange
        sent_events = []

        async def capture(stream, event_json):
            sent_events.append(json.loads(event_json))

        mocker.patch.object(sonic, "_send_event", side_effect=capture)

        # Act
        await sonic._send_audio_chunk(mock_stream, "my-prompt", "my-content", b"\x00")

        # Assert
        audio_input = sent_events[0]["event"]["audioInput"]
        assert audio_input["promptName"] == "my-prompt"
        assert audio_input["contentName"] == "my-content"


# ── _send_session_end ─────────────────────────────────────────────────────────

class TestSendSessionEnd:
    """_send_session_end sends closing events then closes the input stream."""

    @pytest.mark.asyncio
    async def test_sends_content_end_prompt_end_and_session_end(self, mocker, sonic, mock_stream):
        # Arrange
        sent_events = []

        async def capture(stream, event_json):
            sent_events.append(json.loads(event_json))

        mocker.patch.object(sonic, "_send_event", side_effect=capture)

        # Act
        await sonic._send_session_end(mock_stream, "p-name", "a-name")

        # Assert
        event_keys = [list(e["event"].keys())[0] for e in sent_events]
        assert "contentEnd" in event_keys
        assert "promptEnd" in event_keys
        assert "sessionEnd" in event_keys

    @pytest.mark.asyncio
    async def test_closes_input_stream(self, mocker, sonic, mock_stream):
        # Arrange
        mocker.patch.object(sonic, "_send_event", new=mocker.AsyncMock())

        # Act
        await sonic._send_session_end(mock_stream, "p-name", "a-name")

        # Assert
        mock_stream.input_stream.close.assert_awaited_once()


# ── _collect_transcript ───────────────────────────────────────────────────────

class TestCollectTranscript:
    """_collect_transcript accumulates USER text and stops at completionEnd."""

    @pytest.mark.asyncio
    async def test_collects_user_role_text_chunks(self, mocker, sonic):
        # Arrange
        mock_stream = mocker.AsyncMock()
        call_count = 0

        async def fake_await_output():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (None, _response_with_event(mocker, {"contentStart": {"role": "USER"}}))
            elif call_count == 2:
                return (None, _response_with_event(mocker, {"textOutput": {"content": "Hello doctor"}}))
            else:
                return (None, _response_with_event(mocker, {"completionEnd": {}}))

        mock_stream.await_output = fake_await_output

        # Act
        result = await sonic._collect_transcript(mock_stream)

        # Assert
        assert "Hello doctor" in result

    @pytest.mark.asyncio
    async def test_ignores_assistant_role_text(self, mocker, sonic):
        # Arrange
        mock_stream = mocker.AsyncMock()
        call_count = 0

        async def fake_await_output():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (None, _response_with_event(mocker, {"contentStart": {"role": "ASSISTANT"}}))
            elif call_count == 2:
                return (None, _response_with_event(mocker, {"textOutput": {"content": "I am Sonic"}}))
            else:
                return (None, _response_with_event(mocker, {"completionEnd": {}}))

        mock_stream.await_output = fake_await_output

        # Act
        result = await sonic._collect_transcript(mock_stream)

        # Assert
        assert "I am Sonic" not in result
        assert result == ""

    @pytest.mark.asyncio
    async def test_stops_at_completion_end_ignores_later_events(self, mocker, sonic):
        # Arrange
        mock_stream = mocker.AsyncMock()
        call_count = 0

        async def fake_await_output():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (None, _response_with_event(mocker, {"contentStart": {"role": "USER"}}))
            elif call_count == 2:
                return (None, _response_with_event(mocker, {"textOutput": {"content": "First chunk"}}))
            elif call_count == 3:
                return (None, _response_with_event(mocker, {"completionEnd": {}}))
            else:
                return (None, _response_with_event(mocker, {"textOutput": {"content": "After end"}}))

        mock_stream.await_output = fake_await_output

        # Act
        result = await sonic._collect_transcript(mock_stream)

        # Assert
        assert "After end" not in result

    @pytest.mark.asyncio
    async def test_calls_on_chunk_callback_for_each_user_text(self, mocker, sonic):
        # Arrange
        mock_stream = mocker.AsyncMock()
        call_count = 0
        received_chunks = []

        async def on_chunk(text):
            received_chunks.append(text)

        async def fake_await_output():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (None, _response_with_event(mocker, {"contentStart": {"role": "USER"}}))
            elif call_count == 2:
                return (None, _response_with_event(mocker, {"textOutput": {"content": "chunk one"}}))
            elif call_count == 3:
                return (None, _response_with_event(mocker, {"textOutput": {"content": "chunk two"}}))
            else:
                return (None, _response_with_event(mocker, {"completionEnd": {}}))

        mock_stream.await_output = fake_await_output

        # Act
        await sonic._collect_transcript(mock_stream, on_chunk=on_chunk)

        # Assert
        assert received_chunks == ["chunk one", "chunk two"]

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_no_user_text(self, mocker, sonic):
        # Arrange
        mock_stream = mocker.AsyncMock()

        async def fake_await_output():
            return (None, _response_with_event(mocker, {"completionEnd": {}}))

        mock_stream.await_output = fake_await_output

        # Act
        result = await sonic._collect_transcript(mock_stream)

        # Assert
        assert result == ""

    @pytest.mark.asyncio
    async def test_handles_stop_async_iteration_gracefully(self, mocker, sonic):
        # Arrange
        mock_stream = mocker.AsyncMock()

        async def fake_await_output():
            raise StopAsyncIteration

        mock_stream.await_output = fake_await_output

        # Act / Assert — must not raise
        result = await sonic._collect_transcript(mock_stream)
        assert result == ""


# ── _decode_to_pcm16 ──────────────────────────────────────────────────────────

class TestDecodeToPcm16:
    """_decode_to_pcm16 delegates to pydub with correct audio settings.

    _decode_to_pcm16 does a local import: `from pydub import AudioSegment`
    so we must patch 'pydub.AudioSegment', not the sonic module path.
    """

    def test_returns_raw_pcm_bytes(self, mocker, sonic):
        # Arrange
        mock_audio = _make_mock_audio_segment(mocker)
        mocker.patch("pydub.AudioSegment.from_file", return_value=mock_audio)

        # Act
        result = sonic._decode_to_pcm16(b"fake audio bytes")

        # Assert
        assert result == mock_audio.raw_data

    def test_sets_correct_sample_rate(self, mocker, sonic):
        # Arrange
        mock_audio = _make_mock_audio_segment(mocker)
        mocker.patch("pydub.AudioSegment.from_file", return_value=mock_audio)

        # Act
        sonic._decode_to_pcm16(b"audio")

        # Assert
        mock_audio.set_frame_rate.assert_called_once_with(INPUT_SAMPLE_RATE)

    def test_sets_mono_channel(self, mocker, sonic):
        # Arrange
        mock_audio = _make_mock_audio_segment(mocker)
        mocker.patch("pydub.AudioSegment.from_file", return_value=mock_audio)

        # Act
        sonic._decode_to_pcm16(b"audio")

        # Assert
        mock_audio.set_channels.assert_called_once_with(1)

    def test_sets_16_bit_sample_width(self, mocker, sonic):
        # Arrange
        mock_audio = _make_mock_audio_segment(mocker)
        mocker.patch("pydub.AudioSegment.from_file", return_value=mock_audio)

        # Act
        sonic._decode_to_pcm16(b"audio")

        # Assert — 16-bit = 2 bytes sample width
        mock_audio.set_sample_width.assert_called_once_with(2)


# ── transcribe_bytes ──────────────────────────────────────────────────────────

class TestTranscribeBytes:
    """transcribe_bytes: happy path, errors, default system prompt."""

    @pytest.mark.asyncio
    async def test_returns_transcript_on_success(self, mocker, sonic):
        # Arrange
        mocker.patch.object(sonic, "_decode_to_pcm16", return_value=b"\x00\x01" * 512)
        mock_stream = mocker.AsyncMock()
        mock_stream.input_stream = mocker.AsyncMock()
        mocker.patch.object(sonic, "_open_stream", return_value=mock_stream)
        mocker.patch.object(sonic, "_send_session_start", new=mocker.AsyncMock())
        mocker.patch.object(sonic, "_send_prompt_start", new=mocker.AsyncMock())
        mocker.patch.object(sonic, "_send_audio_content_start", new=mocker.AsyncMock())
        mocker.patch.object(sonic, "_send_audio_chunk", new=mocker.AsyncMock())
        mocker.patch.object(sonic, "_send_session_end", new=mocker.AsyncMock())
        mocker.patch.object(sonic, "_collect_transcript", return_value="Doctor: hello Patient: hi")

        # Act
        result = await sonic.transcribe_bytes(b"fake audio")

        # Assert
        assert result["success"] is True
        assert result["transcript"] == "Doctor: hello Patient: hi"

    @pytest.mark.asyncio
    async def test_fires_session_start_prompt_start_and_audio_start(self, mocker, sonic):
        # Arrange
        mocker.patch.object(sonic, "_decode_to_pcm16", return_value=b"\x00" * (CHUNK_SIZE * 2))
        mock_stream = mocker.AsyncMock()
        mock_stream.input_stream = mocker.AsyncMock()
        mocker.patch.object(sonic, "_open_stream", return_value=mock_stream)
        mock_session_start = mocker.patch.object(sonic, "_send_session_start", new=mocker.AsyncMock())
        mock_prompt_start = mocker.patch.object(sonic, "_send_prompt_start", new=mocker.AsyncMock())
        mock_audio_start = mocker.patch.object(sonic, "_send_audio_content_start", new=mocker.AsyncMock())
        mocker.patch.object(sonic, "_send_audio_chunk", new=mocker.AsyncMock())
        mocker.patch.object(sonic, "_send_session_end", new=mocker.AsyncMock())
        mocker.patch.object(sonic, "_collect_transcript", return_value="ok")

        # Act
        await sonic.transcribe_bytes(b"audio")

        # Assert
        mock_session_start.assert_awaited_once()
        mock_prompt_start.assert_awaited_once()
        mock_audio_start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_failure_on_decode_error(self, mocker, sonic):
        # Arrange
        mocker.patch.object(sonic, "_decode_to_pcm16", side_effect=RuntimeError("pydub failed"))

        # Act
        result = await sonic.transcribe_bytes(b"bad audio")

        # Assert
        assert result["success"] is False
        assert "pydub failed" in result["error"]
        assert result["transcript"] == ""

    @pytest.mark.asyncio
    async def test_returns_failure_on_stream_open_error(self, mocker, sonic):
        # Arrange
        mocker.patch.object(sonic, "_decode_to_pcm16", return_value=b"\x00" * 100)
        mocker.patch.object(sonic, "_open_stream", side_effect=ConnectionError("stream failed"))

        # Act
        result = await sonic.transcribe_bytes(b"audio")

        # Assert
        assert result["success"] is False
        assert "stream failed" in result["error"]

    @pytest.mark.asyncio
    async def test_default_system_prompt_is_non_empty_and_mentions_transcribe(self, mocker, sonic):
        # Arrange
        mocker.patch.object(sonic, "_decode_to_pcm16", return_value=b"\x00" * 100)
        mock_stream = mocker.AsyncMock()
        mock_stream.input_stream = mocker.AsyncMock()
        mocker.patch.object(sonic, "_open_stream", return_value=mock_stream)
        mocker.patch.object(sonic, "_send_session_start", new=mocker.AsyncMock())
        mock_prompt_start = mocker.patch.object(sonic, "_send_prompt_start", new=mocker.AsyncMock())
        mocker.patch.object(sonic, "_send_audio_content_start", new=mocker.AsyncMock())
        mocker.patch.object(sonic, "_send_audio_chunk", new=mocker.AsyncMock())
        mocker.patch.object(sonic, "_send_session_end", new=mocker.AsyncMock())
        mocker.patch.object(sonic, "_collect_transcript", return_value="")

        # Act
        await sonic.transcribe_bytes(b"audio", system_prompt=None)

        # Assert — 4th positional arg to _send_prompt_start is the system prompt
        system_arg = mock_prompt_start.call_args[0][3]
        assert len(system_arg) > 0
        assert "transcrib" in system_arg.lower()


# ── transcribe_stream ─────────────────────────────────────────────────────────

class TestTranscribeStream:
    """transcribe_stream: happy path, empty buffer, decode failure, chunk accumulation."""

    @pytest.mark.asyncio
    async def test_yields_error_when_no_audio_chunks_provided(self, mocker, sonic):
        # Arrange
        async def empty_gen():
            return
            yield  # mark as async generator

        # Act
        results = []
        async for event in sonic.transcribe_stream(empty_gen()):
            results.append(event)

        # Assert
        assert len(results) == 1
        assert results[0]["type"] == "error"
        assert results[0]["final"] is True

    @pytest.mark.asyncio
    async def test_yields_error_when_decode_fails(self, mocker, sonic):
        # Arrange
        async def audio_gen():
            yield b"\x00\x01"

        mocker.patch.object(sonic, "_decode_to_pcm16", side_effect=RuntimeError("decode failed"))

        # Act
        results = []
        async for event in sonic.transcribe_stream(audio_gen()):
            results.append(event)

        # Assert
        assert results[-1]["type"] == "error"
        assert "decode failed" in results[-1]["error"]
        assert results[-1]["final"] is True

    @pytest.mark.asyncio
    async def test_accumulates_all_chunks_before_decoding(self, mocker, sonic):
        # Arrange — decode will raise to stop execution early; we only care
        # that it was called once with ALL chunks concatenated
        chunks_seen = []

        def capture_decode(audio_bytes):
            chunks_seen.append(audio_bytes)
            raise RuntimeError("stop after decode")

        async def audio_gen():
            yield b"\x00\x01"
            yield b"\x02\x03"
            yield b"\x04\x05"

        mocker.patch.object(sonic, "_decode_to_pcm16", side_effect=capture_decode)

        # Act
        results = []
        async for event in sonic.transcribe_stream(audio_gen()):
            results.append(event)

        # Assert — decode called exactly once with all three chunks joined
        assert len(chunks_seen) == 1
        assert chunks_seen[0] == b"\x00\x01\x02\x03\x04\x05"

    @pytest.mark.asyncio
    async def test_final_event_always_has_final_true(self, mocker, sonic):
        # Arrange
        async def audio_gen():
            yield b"\x00\x01"

        mocker.patch.object(sonic, "_decode_to_pcm16", side_effect=RuntimeError("stop"))

        # Act
        results = []
        async for event in sonic.transcribe_stream(audio_gen()):
            results.append(event)

        # Assert
        assert results[-1]["final"] is True

    @pytest.mark.asyncio
    async def test_yields_error_on_stream_open_failure(self, mocker, sonic):
        # Arrange
        async def audio_gen():
            yield b"\x00\x01"

        mocker.patch.object(sonic, "_decode_to_pcm16", return_value=b"\x00" * 64)
        mocker.patch.object(sonic, "_open_stream", side_effect=ConnectionError("no stream"))

        # Act
        results = []
        async for event in sonic.transcribe_stream(audio_gen()):
            results.append(event)

        # Assert
        assert results[-1]["type"] == "error"
        assert "no stream" in results[-1]["error"]