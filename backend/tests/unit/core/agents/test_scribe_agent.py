"""
Unit tests for ScribeAgent.

Covers:
- Initialization: dependency wiring
- process_message: delegates to process_consultation
- process_consultation: tool injection, LLM loop, cache reads, exception handling
- approve_and_insert: EHR insert via PatientTools, missing tool fallback
- process_audio: audio reading, delegation
- process_real_time_audio: delegation to transcription service
- end_real_time_session: transcript assembly, consultation call, cleanup
"""
import pytest
from src.core.agents.scribe import ScribeAgent


SAMPLE_TRANSCRIPT = "Doctor: How are you? Patient: Not great."
SAMPLE_SOAP = {"subjective": "Not great", "objective": {}, "assessment": "", "plan": ""}
SAMPLE_SCORES = {"completeness": 85, "hallucination_risk": "low", "drug_safety": 100, "overall_ready": True}
SAMPLE_PATIENT = {"patient_id": "P001", "name": "John Smith", "conditions": ["HTN"]}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_model(mocker):
    model = mocker.Mock()
    model.tool_registry = None

    async def _fake_prompt(*args, **kwargs):
        yield {"text": "scribe complete"}

    model.prompt = mocker.Mock(return_value=_fake_prompt())
    return model


@pytest.fixture
def mock_tool_registry(mocker):
    registry = mocker.Mock()
    registry.get_available_tools = mocker.Mock(return_value=["get_patient_history", "generate_soap_note"])
    registry.tool_classes = [mocker.Mock(kwargs={}), mocker.Mock(kwargs={})]
    return registry


@pytest.fixture
def mock_prompt_template(mocker):
    template = mocker.Mock()
    template.get_system_prompt = mocker.Mock(return_value="You are MedScribe.")
    return template


@pytest.fixture
def mock_transcription_service(mocker):
    service = mocker.Mock()
    service.process_real_time_audio = mocker.AsyncMock()
    service.get_accumulated_transcript = mocker.AsyncMock(return_value=SAMPLE_TRANSCRIPT)
    service.clear_realtime_transcript = mocker.AsyncMock()
    return service


@pytest.fixture
def mock_soap_service(mocker):
    service = mocker.Mock()
    service.get_session_transcript = mocker.AsyncMock(return_value={"transcript": SAMPLE_TRANSCRIPT})
    return service


@pytest.fixture
def mock_evaluation_service(mocker):
    return mocker.Mock()


@pytest.fixture
def mock_cache(mocker):
    cache = mocker.Mock()

    async def _get(key):
        if "soap:" in key:
            return SAMPLE_SOAP
        if "evaluation:scores:" in key:
            return SAMPLE_SCORES
        if "patient:context:" in key:
            return SAMPLE_PATIENT
        return None

    cache.get = mocker.AsyncMock(side_effect=_get)
    return cache


@pytest.fixture
def scribe_agent(
    mock_llm_model, mock_tool_registry, mock_prompt_template,
    mock_transcription_service, mock_soap_service, mock_evaluation_service, mock_cache
):
    return ScribeAgent(
        llm_model=mock_llm_model,
        tool_registry=mock_tool_registry,
        prompt_template=mock_prompt_template,
        transcription_service=mock_transcription_service,
        soap_service=mock_soap_service,
        evaluation_service=mock_evaluation_service,
        cache_service=mock_cache,
    )


# ── Initialization ────────────────────────────────────────────────────────────

class TestScribeAgentInitialization:
    """Agent wires dependencies correctly on construction."""

    def test_sets_tool_registry_on_llm_model(
        self, mock_llm_model, mock_tool_registry, mock_prompt_template,
        mock_transcription_service, mock_soap_service, mock_evaluation_service, mock_cache
    ):
        # Act
        ScribeAgent(
            llm_model=mock_llm_model,
            tool_registry=mock_tool_registry,
            prompt_template=mock_prompt_template,
            transcription_service=mock_transcription_service,
            soap_service=mock_soap_service,
            evaluation_service=mock_evaluation_service,
            cache_service=mock_cache,
        )

        # Assert
        assert mock_llm_model.tool_registry == mock_tool_registry

    def test_queries_available_tools_on_init(
        self, mock_llm_model, mock_tool_registry, mock_prompt_template,
        mock_transcription_service, mock_soap_service, mock_evaluation_service, mock_cache
    ):
        # Act
        ScribeAgent(
            llm_model=mock_llm_model,
            tool_registry=mock_tool_registry,
            prompt_template=mock_prompt_template,
            transcription_service=mock_transcription_service,
            soap_service=mock_soap_service,
            evaluation_service=mock_evaluation_service,
            cache_service=mock_cache,
        )

        # Assert
        mock_tool_registry.get_available_tools.assert_called_once()

    def test_handles_none_tool_registry_without_error(
        self, mock_llm_model, mock_prompt_template,
        mock_transcription_service, mock_soap_service, mock_evaluation_service, mock_cache
    ):
        # Act
        agent = ScribeAgent(
            llm_model=mock_llm_model,
            tool_registry=None,
            prompt_template=mock_prompt_template,
            transcription_service=mock_transcription_service,
            soap_service=mock_soap_service,
            evaluation_service=mock_evaluation_service,
            cache_service=mock_cache,
        )

        # Assert
        assert agent.tool_registry is None


# ── process_message ───────────────────────────────────────────────────────────

class TestProcessMessage:
    """process_message is a thin wrapper around process_consultation."""

    @pytest.mark.asyncio
    async def test_delegates_to_process_consultation(self, mocker, scribe_agent):
        # Arrange
        scribe_agent.process_consultation = mocker.AsyncMock(
            return_value={"success": True, "session_id": "s1"}
        )

        # Act
        result = await scribe_agent.process_message(message=SAMPLE_TRANSCRIPT)

        # Assert
        scribe_agent.process_consultation.assert_awaited_once_with(transcript=SAMPLE_TRANSCRIPT)
        assert result["success"] is True


# ── process_consultation: tool kwargs injection ───────────────────────────────

class TestProcessConsultationToolInjection:
    """Tool kwargs are populated with session context before the LLM loop."""

    @pytest.mark.asyncio
    async def test_injects_patient_id_session_id_and_transcript(
        self, scribe_agent, mock_tool_registry
    ):
        # Act
        await scribe_agent.process_consultation(
            transcript=SAMPLE_TRANSCRIPT,
            patient_id="P042",
            session_id="inject-sess",
        )

        # Assert
        for tool in mock_tool_registry.tool_classes:
            assert tool.kwargs["patient_id"] == "P042"
            assert tool.kwargs["session_id"] == "inject-sess"
            assert tool.kwargs["transcript"] == SAMPLE_TRANSCRIPT

    @pytest.mark.asyncio
    async def test_initial_soap_and_conditions_are_empty_sentinels(
        self, scribe_agent, mock_tool_registry
    ):
        # Act
        await scribe_agent.process_consultation(
            transcript=SAMPLE_TRANSCRIPT,
            session_id="sentinel-sess",
        )

        # Assert
        for tool in mock_tool_registry.tool_classes:
            assert tool.kwargs["soap_note"] == {}
            assert tool.kwargs["conditions"] == []
            assert tool.kwargs["patient_context"] is None

    @pytest.mark.asyncio
    async def test_skips_injection_when_no_registry(
        self, mocker, mock_llm_model, mock_prompt_template,
        mock_transcription_service, mock_soap_service, mock_evaluation_service, mock_cache
    ):
        # Arrange
        agent = ScribeAgent(
            llm_model=mock_llm_model,
            tool_registry=None,
            prompt_template=mock_prompt_template,
            transcription_service=mock_transcription_service,
            soap_service=mock_soap_service,
            evaluation_service=mock_evaluation_service,
            cache_service=mock_cache,
        )

        # Act / Assert — no exception raised
        result = await agent.process_consultation(transcript=SAMPLE_TRANSCRIPT)
        assert result["success"] is True


# ── process_consultation: LLM loop ───────────────────────────────────────────

class TestProcessConsultationLLMLoop:
    """LLM.prompt is called with the correct parameters."""

    @pytest.mark.asyncio
    async def test_calls_llm_prompt_once(self, scribe_agent, mock_llm_model):
        # Act
        await scribe_agent.process_consultation(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        mock_llm_model.prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_called_with_stream_false_and_tools_enabled(self, scribe_agent, mock_llm_model):
        # Act
        await scribe_agent.process_consultation(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        call_kwargs = mock_llm_model.prompt.call_args[1]
        assert call_kwargs["stream"] is False
        assert call_kwargs["enable_tools"] is True

    @pytest.mark.asyncio
    async def test_transcript_present_in_llm_text_argument(self, scribe_agent, mock_llm_model):
        # Act
        await scribe_agent.process_consultation(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        text_arg = mock_llm_model.prompt.call_args[1]["text"]
        assert SAMPLE_TRANSCRIPT in text_arg

    @pytest.mark.asyncio
    async def test_system_prompt_generated_with_current_date(self, scribe_agent, mock_prompt_template):
        # Act
        await scribe_agent.process_consultation(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        mock_prompt_template.get_system_prompt.assert_called_once()
        call_kwargs = mock_prompt_template.get_system_prompt.call_args[1]
        assert "current_date" in call_kwargs


# ── process_consultation: cache reads ────────────────────────────────────────

class TestProcessConsultationCacheReads:
    """Results are assembled from cache after the LLM loop completes."""

    @pytest.mark.asyncio
    async def test_returns_soap_from_cache(self, scribe_agent):
        # Act
        result = await scribe_agent.process_consultation(
            transcript=SAMPLE_TRANSCRIPT, session_id="soap-sess"
        )

        # Assert
        assert result["soap"] == SAMPLE_SOAP

    @pytest.mark.asyncio
    async def test_returns_scores_from_cache(self, scribe_agent):
        # Act
        result = await scribe_agent.process_consultation(
            transcript=SAMPLE_TRANSCRIPT, session_id="score-sess"
        )

        # Assert
        assert result["scores"] == SAMPLE_SCORES

    @pytest.mark.asyncio
    async def test_returns_patient_context_from_cache(self, scribe_agent):
        # Act
        result = await scribe_agent.process_consultation(
            transcript=SAMPLE_TRANSCRIPT, patient_id="P001", session_id="ctx-sess"
        )

        # Assert
        assert result["patient_context"] == SAMPLE_PATIENT

    @pytest.mark.asyncio
    async def test_returns_empty_dicts_on_cache_miss(self, mocker, scribe_agent, mock_cache):
        # Arrange — all cache reads return None
        mock_cache.get = mocker.AsyncMock(return_value=None)

        # Act
        result = await scribe_agent.process_consultation(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        assert result["soap"] == {}
        assert result["scores"] == {}
        assert result["patient_context"] == {}

    @pytest.mark.asyncio
    async def test_result_contains_all_required_top_level_keys(self, scribe_agent):
        # Act
        result = await scribe_agent.process_consultation(
            transcript=SAMPLE_TRANSCRIPT, session_id="keys-sess"
        )

        # Assert
        for key in ("success", "session_id", "soap", "scores", "patient_context", "missing_fields", "transcript"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_echoes_original_transcript_in_result(self, scribe_agent):
        # Act
        result = await scribe_agent.process_consultation(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        assert result["transcript"] == SAMPLE_TRANSCRIPT

    @pytest.mark.asyncio
    async def test_auto_generates_session_id_when_not_provided(self, scribe_agent):
        # Act
        result = await scribe_agent.process_consultation(transcript=SAMPLE_TRANSCRIPT)

        # Assert — UUID4 is 36 chars
        assert "session_id" in result
        assert len(result["session_id"]) == 36


# ── process_consultation: exception handling ──────────────────────────────────

class TestProcessConsultationExceptionHandling:
    """Exceptions are caught and returned as a failure dict."""

    @pytest.mark.asyncio
    async def test_returns_failure_on_llm_crash(
        self, mocker, mock_llm_model, mock_tool_registry, mock_prompt_template,
        mock_transcription_service, mock_soap_service, mock_evaluation_service, mock_cache
    ):
        # Arrange
        mock_llm_model.prompt = mocker.Mock(side_effect=RuntimeError("LLM crashed"))
        agent = ScribeAgent(
            llm_model=mock_llm_model,
            tool_registry=mock_tool_registry,
            prompt_template=mock_prompt_template,
            transcription_service=mock_transcription_service,
            soap_service=mock_soap_service,
            evaluation_service=mock_evaluation_service,
            cache_service=mock_cache,
        )

        # Act
        result = await agent.process_consultation(
            transcript=SAMPLE_TRANSCRIPT, session_id="crash-sess"
        )

        # Assert
        assert result["success"] is False
        assert "LLM crashed" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_failure_on_cache_crash(self, mocker, scribe_agent, mock_cache):
        # Arrange
        mock_cache.get = mocker.AsyncMock(side_effect=ConnectionError("Redis down"))

        # Act
        result = await scribe_agent.process_consultation(transcript=SAMPLE_TRANSCRIPT)

        # Assert
        assert result["success"] is False
        assert "Redis down" in result["error"]

    @pytest.mark.asyncio
    async def test_session_id_always_in_failure_result(
        self, mocker, mock_llm_model, mock_tool_registry, mock_prompt_template,
        mock_transcription_service, mock_soap_service, mock_evaluation_service, mock_cache
    ):
        # Arrange
        mock_llm_model.prompt = mocker.Mock(side_effect=ValueError("oops"))
        agent = ScribeAgent(
            llm_model=mock_llm_model,
            tool_registry=mock_tool_registry,
            prompt_template=mock_prompt_template,
            transcription_service=mock_transcription_service,
            soap_service=mock_soap_service,
            evaluation_service=mock_evaluation_service,
            cache_service=mock_cache,
        )
        session_id = "must-survive"

        # Act
        result = await agent.process_consultation(
            transcript=SAMPLE_TRANSCRIPT, session_id=session_id
        )

        # Assert
        assert result["session_id"] == session_id



# ── process_audio ─────────────────────────────────────────────────────────────

class TestProcessAudio:
    """process_audio reads bytes then delegates to process_consultation."""

    @pytest.mark.asyncio
    async def test_reads_audio_bytes_from_upload(self, mocker, scribe_agent):
        # Arrange
        audio_mock = mocker.Mock()
        audio_mock.read = mocker.AsyncMock(return_value=b"raw audio bytes")
        scribe_agent.process_consultation = mocker.AsyncMock(return_value={"success": True})

        # Act
        await scribe_agent.process_audio(audio=audio_mock)

        # Assert
        audio_mock.read.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_failure_when_audio_read_fails(self, mocker, scribe_agent):
        # Arrange
        audio_mock = mocker.Mock()
        audio_mock.read = mocker.AsyncMock(side_effect=IOError("disk error"))

        # Act
        result = await scribe_agent.process_audio(audio=audio_mock, session_id="audio-fail")

        # Assert
        assert result["success"] is False
        assert "disk error" in result["error"]


# ── process_real_time_audio ───────────────────────────────────────────────────

class TestProcessRealTimeAudio:
    """process_real_time_audio delegates directly to the transcription service."""

    @pytest.mark.asyncio
    async def test_delegates_chunk_to_transcription_service(self, mocker, scribe_agent, mock_transcription_service):
        # Arrange
        send_fn = mocker.AsyncMock()

        # Act
        await scribe_agent.process_real_time_audio(
            audio_bytes=b"chunk",
            patient_id="P001",
            session_id="rt-sess",
            send_to_socket=send_fn,
        )

        # Assert
        mock_transcription_service.process_real_time_audio.assert_awaited_once_with(
            audio_bytes=b"chunk",
            patient_id="P001",
            send_to_socket=send_fn,
        )


# ── end_real_time_session ─────────────────────────────────────────────────────

class TestEndRealTimeSession:
    """end_real_time_session assembles the full transcript then runs process_consultation."""

    @pytest.mark.asyncio
    async def test_retrieves_accumulated_transcript(self, mocker, scribe_agent, mock_transcription_service):
        # Arrange
        scribe_agent.process_consultation = mocker.AsyncMock(return_value={"success": True})

        # Act
        await scribe_agent.end_real_time_session(patient_id="P001", session_id="rt-sess")

        # Assert
        mock_transcription_service.get_accumulated_transcript.assert_awaited_once_with("P001")

    @pytest.mark.asyncio
    async def test_calls_process_consultation_with_assembled_transcript(
        self, mocker, scribe_agent, mock_transcription_service
    ):
        # Arrange
        scribe_agent.process_consultation = mocker.AsyncMock(return_value={"success": True})

        # Act
        await scribe_agent.end_real_time_session(patient_id="P001", session_id="rt-sess")

        # Assert
        scribe_agent.process_consultation.assert_awaited_once_with(
            transcript=SAMPLE_TRANSCRIPT,
            patient_id="P001",
            session_id="rt-sess",
        )

    @pytest.mark.asyncio
    async def test_clears_transcript_cache_after_processing(self, mocker, scribe_agent, mock_transcription_service):
        # Arrange
        scribe_agent.process_consultation = mocker.AsyncMock(return_value={"success": True})

        # Act
        await scribe_agent.end_real_time_session(patient_id="P001", session_id="clear-sess")

        # Assert
        mock_transcription_service.clear_realtime_transcript.assert_awaited_once_with("P001")

    @pytest.mark.asyncio
    async def test_uses_demo_fallback_when_no_transcript_accumulated(
        self, mocker, scribe_agent, mock_transcription_service
    ):
        # Arrange
        mock_transcription_service.get_accumulated_transcript = mocker.AsyncMock(return_value=None)
        scribe_agent.process_consultation = mocker.AsyncMock(return_value={"success": True})

        # Act
        await scribe_agent.end_real_time_session(patient_id="P001", session_id="demo-sess")

        # Assert — consultation still fired with a non-empty transcript
        scribe_agent.process_consultation.assert_awaited_once()
        call_kwargs = scribe_agent.process_consultation.call_args[1]
        assert call_kwargs["transcript"]  # not empty

    @pytest.mark.asyncio
    async def test_returns_result_from_process_consultation(self, mocker, scribe_agent):
        # Arrange
        expected = {"success": True, "session_id": "rt-ret", "soap": SAMPLE_SOAP}
        scribe_agent.process_consultation = mocker.AsyncMock(return_value=expected)

        # Act
        result = await scribe_agent.end_real_time_session(patient_id="P001", session_id="rt-ret")

        # Assert
        assert result == expected