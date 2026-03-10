"""
AWS Nova 2 Sonic language model implementation.

Mirrors BedrockModel structure but is purpose-built for audio input.
Supports two modes:
  - transcribe_bytes()  : complete audio bytes at once (file upload path)
  - transcribe_stream() : real-time audio chunk streaming (WebSocket path)

Does NOT handle SOAP generation, reasoning, or any LLM text tasks.
Those stay in BedrockModel. This model's only job is audio → transcript.
"""
import asyncio
import base64
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import Config
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

from utils.logger import get_logger

logger = get_logger()

# ── Audio constants (match Nova 2 Sonic requirements) ─────────────────────────
INPUT_SAMPLE_RATE = 16000   # Hz — required by Sonic
INPUT_SAMPLE_BITS = 16
INPUT_CHANNELS = 1
CHUNK_SIZE = 1024           # frames per buffer


class SonicModel:
    """
    Nova 2 Sonic model wrapper for audio transcription.

    Mirrors BedrockModel in structure:
      - __init__ takes credentials and config
      - _initialize_client() creates a fresh client per call (NOT cached —
        bidirectional streaming clients cannot be reused across sessions)
      - Public methods are the two transcription modes

    Does NOT use aioboto3 / converse API — Sonic requires the
    bidirectional streaming SDK (aws_sdk_bedrock_runtime).
    """

    def __init__(
        self,
        aws_access_key: str,
        aws_secret_key: str,
        region_name: str = "us-east-1",
        model_id: str = "amazon.nova-2-sonic-v1:0",
    ):
        """
        Initialize SonicModel.

        Args:
            aws_access_key: AWS access key ID
            aws_secret_key: AWS secret access key
            region_name: AWS region (Sonic is us-east-1 only currently)
            model_id: Sonic model identifier
        """
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.region_name = region_name
        self.model_id = model_id

        # Set env vars so EnvironmentCredentialsResolver picks them up
        import os
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
        os.environ["AWS_DEFAULT_REGION"] = region_name

    # ── Client ────────────────────────────────────────────────────────────────

    def _initialize_client(self) -> BedrockRuntimeClient:
        """
        Create a fresh Sonic bidirectional streaming client.

        NOT cached — the SDK client cannot be reused across separate
        bidirectional stream sessions. Reusing causes ValidationException
        on the second and subsequent calls.
        """
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region_name}.amazonaws.com",
            region=self.region_name,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        return BedrockRuntimeClient(config=config)

    # ── Session helpers ───────────────────────────────────────────────────────

    async def _open_stream(self) -> Any:
        """Open a bidirectional stream to Sonic. Always uses a fresh client."""
        client = self._initialize_client()
        logger.info(f"SonicModel: opening new bidirectional stream ({self.model_id})")
        stream = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        return stream

    async def _send_event(self, stream: Any, event_json: str) -> None:
        """Send a raw JSON event to the open Sonic stream."""
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await stream.input_stream.send(event)

    async def _send_session_start(self, stream: Any) -> None:
        """Send sessionStart event — required first event in every session."""
        await self._send_event(stream, json.dumps({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 4096,
                        "topP": 0.9,
                        "temperature": 0.2,
                    }
                }
            }
        }))

    async def _send_prompt_start(
        self,
        stream: Any,
        prompt_name: str,
        content_name: str,
        system_prompt: str,
    ) -> None:
        """Send promptStart + system prompt events."""
        # audioOutputConfiguration is REQUIRED by Sonic even though we never
        # use the audio output — omitting it causes ValidationException.
        await self._send_event(stream, json.dumps({
            "event": {
                "promptStart": {
                    "promptName": prompt_name,
                    "textOutputConfiguration": {
                        "mediaType": "text/plain"
                    },
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "matthew",
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                }
            }
        }))

        # System prompt content block
        sys_content_name = str(uuid.uuid4())
        await self._send_event(stream, json.dumps({
            "event": {
                "contentStart": {
                    "promptName": prompt_name,
                    "contentName": sys_content_name,
                    "type": "TEXT",
                    "interactive": False,
                    "role": "SYSTEM",
                    "textInputConfiguration": {"mediaType": "text/plain"}
                }
            }
        }))
        await self._send_event(stream, json.dumps({
            "event": {
                "textInput": {
                    "promptName": prompt_name,
                    "contentName": sys_content_name,
                    "content": system_prompt,
                }
            }
        }))
        await self._send_event(stream, json.dumps({
            "event": {
                "contentEnd": {
                    "promptName": prompt_name,
                    "contentName": sys_content_name,
                }
            }
        }))

    async def _send_audio_content_start(
        self,
        stream: Any,
        prompt_name: str,
        audio_content_name: str,
    ) -> None:
        """Open the audio input content block."""
        await self._send_event(stream, json.dumps({
            "event": {
                "contentStart": {
                    "promptName": prompt_name,
                    "contentName": audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": INPUT_SAMPLE_RATE,
                        "sampleSizeBits": INPUT_SAMPLE_BITS,
                        "channelCount": INPUT_CHANNELS,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    }
                }
            }
        }))

    async def _send_audio_chunk(
        self,
        stream: Any,
        prompt_name: str,
        audio_content_name: str,
        audio_bytes: bytes,
    ) -> None:
        """Send a single audio chunk to Sonic."""
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        await self._send_event(stream, json.dumps({
            "event": {
                "audioInput": {
                    "promptName": prompt_name,
                    "contentName": audio_content_name,
                    "content": encoded,
                }
            }
        }))

    async def _send_session_end(
        self,
        stream: Any,
        prompt_name: str,
        audio_content_name: str,
    ) -> None:
        """Close audio content block, prompt, and session."""
        await self._send_event(stream, json.dumps({
            "event": {"contentEnd": {"promptName": prompt_name, "contentName": audio_content_name}}
        }))
        await self._send_event(stream, json.dumps({
            "event": {"promptEnd": {"promptName": prompt_name}}
        }))
        await self._send_event(stream, json.dumps({
            "event": {"sessionEnd": {}}
        }))
        await stream.input_stream.close()

    # ── Response collector ────────────────────────────────────────────────────

    async def _collect_transcript(
        self,
        stream: Any,
        on_chunk: Optional[Any] = None,
    ) -> str:
        """
        Read response events from Sonic and collect transcript text.

        Role is tracked from contentStart events (not textOutput).

        USER role textOutput = transcription of the input audio speech.
        ASSISTANT role textOutput = Sonic's own generated response (ignored).
        audioOutput events = Sonic voice audio (ignored — we only want text).

        Args:
            stream: Open Sonic bidirectional stream
            on_chunk: Optional async callable(chunk: str) called for each
                      transcript chunk as it arrives

        Returns:
            Full accumulated transcript string
        """
        transcript_parts: List[str] = []
        current_role: Optional[str] = None

        try:
            while True:
                output = await stream.await_output()
                result = await output[1].receive()

                if not (result.value and result.value.bytes_):
                    continue

                json_data = json.loads(result.value.bytes_.decode("utf-8"))
                event = json_data.get("event", {})

                if "contentStart" in event:
                    current_role = event["contentStart"].get("role")

                elif "textOutput" in event:
                    text = event["textOutput"].get("content", "")
                    if text and current_role == "USER":
                        transcript_parts.append(text)
                        logger.debug(f"Transcript chunk [{current_role}]: {text[:60]}...")
                        if on_chunk:
                            await on_chunk(text)

                elif "audioOutput" in event:
                    pass  # intentionally ignored

                elif "completionEnd" in event:
                    break

        except StopAsyncIteration:
            pass
        # except Exception as e:
        #     logger.error(f"Error collecting transcript: {e}", exc_info=True)
        except Exception as e:
            # ValidationException after completionEnd is Sonic's normal session-close
            # signal — not a real error. Only log if we got no transcript at all.
            err_str = str(e)
            if "Invalid input request" in err_str:
                logger.debug(f"Sonic session closed (expected): {err_str}")
            else:
                logger.error(f"Error collecting transcript: {e}", exc_info=True)

        return " ".join(transcript_parts).strip()

    # ── Audio decoder ─────────────────────────────────────────────────────────

    def _decode_to_pcm16(self, audio_bytes: bytes) -> bytes:
        """
        Decode any audio format to raw PCM16 at 16kHz mono.

        Sonic requires raw PCM16 samples — no file headers, no encoding.
        Uses pydub to handle WAV, MP3, M4A, OGG, WebM/Opus etc.

        Args:
            audio_bytes: Raw bytes of any supported audio file

        Returns:
            Raw PCM16 bytes at 16kHz mono, no header
        """
        try:
            import io
            from pydub import AudioSegment

            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio = audio.set_frame_rate(INPUT_SAMPLE_RATE)
            audio = audio.set_channels(INPUT_CHANNELS)
            audio = audio.set_sample_width(2)  # 16-bit = 2 bytes

            raw_pcm = audio.raw_data
            duration_secs = len(audio) / 1000.0
            logger.info(
                f"Decoded audio: {duration_secs:.1f}s, "
                f"{len(raw_pcm)} PCM bytes @ {INPUT_SAMPLE_RATE}Hz mono"
            )
            return raw_pcm

        except ImportError:
            logger.error("pydub not installed. Run: pip install pydub")
            raise
        except Exception as e:
            logger.error(f"Audio decode failed: {e}", exc_info=True)
            raise

    # ── Public API ────────────────────────────────────────────────────────────

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe complete audio bytes in one shot.

        Used by TranscriptionService for the file upload path.
        Decodes audio to PCM16 first, then sends at natural audio pace
        so Sonic processes it as if it were a real-time stream.

        Args:
            audio_bytes: Raw bytes of any audio file (WAV, MP3, M4A, WebM, etc.)
            system_prompt: Optional instruction for Sonic

        Returns:
            {"success": True, "transcript": "..."} or {"success": False, "error": "..."}
        """
        prompt_name = str(uuid.uuid4())
        audio_content_name = str(uuid.uuid4())

        _system_prompt = system_prompt or (
            "You are transcribing a medical consultation between a doctor and patient. "
            "Transcribe all speech accurately. Do not respond or add commentary."
        )

        try:
            # Decode to raw PCM16 at 16kHz mono before opening stream.
            # Stream is opened only when audio is ready to flow immediately —
            # Sonic times out if a session is opened but audio doesn't arrive promptly.
            pcm_bytes = await asyncio.get_event_loop().run_in_executor(
                None, self._decode_to_pcm16, audio_bytes
            )

            stream = await self._open_stream()
            logger.info(f"SonicModel.transcribe_bytes: session {prompt_name[:8]}")

            await self._send_session_start(stream)
            await self._send_prompt_start(stream, prompt_name, audio_content_name, _system_prompt)
            await self._send_audio_content_start(stream, prompt_name, audio_content_name)

            # Send chunks paced at natural audio rate so Sonic isn't overwhelmed.
            # CHUNK_SIZE samples at 16kHz = 64ms per chunk.
            bytes_per_chunk = CHUNK_SIZE * 2  # 16-bit = 2 bytes per sample
            chunk_duration_s = CHUNK_SIZE / INPUT_SAMPLE_RATE
            sleep_between_chunks = chunk_duration_s * 0.6  # 60% of real-time pace

            total_chunks = 0
            for i in range(0, len(pcm_bytes), bytes_per_chunk):
                chunk = pcm_bytes[i: i + bytes_per_chunk]
                if chunk:
                    await self._send_audio_chunk(stream, prompt_name, audio_content_name, chunk)
                    total_chunks += 1
                    await asyncio.sleep(sleep_between_chunks)

            logger.info(f"Sent {total_chunks} audio chunks")

            # Collect transcript concurrently with session close
            transcript_task = asyncio.create_task(
                self._collect_transcript(stream)
            )
            await self._send_session_end(stream, prompt_name, audio_content_name)

            transcript = await transcript_task

            logger.info(f"Transcription complete: {len(transcript)} chars")
            return {"success": True, "transcript": transcript}

        except Exception as e:
            logger.error(f"transcribe_bytes failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "transcript": ""}

    async def transcribe_stream(
        self,
        audio_chunk_generator: AsyncGenerator[bytes, None],
        on_transcript_chunk: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Transcribe audio in real time from an async chunk generator.

        Used DIRECTLY by the WebSocket route — not through TranscriptionService.
        Yields transcript chunks as Sonic produces them so the frontend
        can display live transcription.

        Args:
            audio_chunk_generator: AsyncGenerator yielding encoded audio bytes
                                   (WebM/Opus from browser MediaRecorder).
                                   All chunks are accumulated and decoded to
                                   PCM16 before the Sonic stream is opened.
            on_transcript_chunk: Optional async callable(chunk: str) —
                                 additional side-effect per chunk if needed
            system_prompt: Optional Sonic system instruction

        Yields:
            {"type": "transcript_chunk", "text": "...", "final": False}
            {"type": "transcript_complete", "transcript": "...", "final": True}
            {"type": "error", "error": "...", "final": True}
        """
        prompt_name = str(uuid.uuid4())
        audio_content_name = str(uuid.uuid4())

        _system_prompt = system_prompt or (
            "You are transcribing a live medical consultation between a doctor and patient. "
            "Transcribe all speech accurately as it happens. Do not respond or add commentary."
        )

        transcript_parts: List[str] = []

        try:
            # ── Step 1: Accumulate + decode BEFORE opening the stream ──────────
            # pydub needs the full WebM/Opus container to decode correctly.
            # Stream is opened only when PCM bytes are ready to send immediately —
            # Sonic times out if a session sits idle after opening.
            webm_buffer = bytearray()
            async for chunk in audio_chunk_generator:
                if chunk:
                    webm_buffer.extend(chunk)

            if not webm_buffer:
                logger.warning("transcribe_stream: no audio data received")
                yield {"type": "error", "error": "No audio data received", "final": True}
                return

            pcm_bytes = await asyncio.get_event_loop().run_in_executor(
                None, self._decode_to_pcm16, bytes(webm_buffer)
            )
            logger.info(
                f"transcribe_stream: decoded {len(pcm_bytes)} PCM bytes, opening stream"
            )

            # ── Step 2: Fresh client + stream ──────────────────────────────────
            # A new client is created for every call — bidirectional streaming
            # clients cannot be reused across sessions (causes ValidationException).
            stream = await self._open_stream()
            logger.info(f"SonicModel.transcribe_stream: session {prompt_name[:8]}")

            await self._send_session_start(stream)
            await self._send_prompt_start(stream, prompt_name, audio_content_name, _system_prompt)
            await self._send_audio_content_start(stream, prompt_name, audio_content_name)

            # ── Step 3: Send audio + collect responses concurrently ─────────────
            # Audio is already decoded — send_audio flows immediately after
            # session init, so Sonic never sees an idle open session.
            async def send_audio():
                bytes_per_chunk = CHUNK_SIZE * 2
                chunk_duration_s = CHUNK_SIZE / INPUT_SAMPLE_RATE
                sleep_between_chunks = chunk_duration_s * 0.6

                for i in range(0, len(pcm_bytes), bytes_per_chunk):
                    chunk = pcm_bytes[i: i + bytes_per_chunk]
                    if chunk:
                        await self._send_audio_chunk(
                            stream, prompt_name, audio_content_name, chunk
                        )
                        await asyncio.sleep(sleep_between_chunks)

                await self._send_session_end(stream, prompt_name, audio_content_name)

            response_queue: asyncio.Queue = asyncio.Queue()

            async def collect_responses():
                current_role: Optional[str] = None
                try:
                    while True:
                        output = await stream.await_output()
                        result = await output[1].receive()

                        if not (result.value and result.value.bytes_):
                            continue

                        json_data = json.loads(result.value.bytes_.decode("utf-8"))
                        event = json_data.get("event", {})

                        if "contentStart" in event:
                            current_role = event["contentStart"].get("role")

                        elif "textOutput" in event:
                            text = event["textOutput"].get("content", "")
                            if text and current_role == "USER":
                                await response_queue.put({"type": "chunk", "text": text})
                            # ASSISTANT text = Sonic response — ignored

                        elif "audioOutput" in event:
                            pass  # intentionally ignored

                        elif "completionEnd" in event:
                            await response_queue.put({"type": "done"})
                            break

                except StopAsyncIteration:
                    await response_queue.put({"type": "done"})
                except Exception as e:
                    logger.error(f"collect_responses error: {e}", exc_info=True)
                    await response_queue.put({"type": "error", "error": str(e)})

            send_task = asyncio.create_task(send_audio())
            collect_task = asyncio.create_task(collect_responses())

            while True:
                item = await response_queue.get()

                if item["type"] == "chunk":
                    text = item["text"]
                    transcript_parts.append(text)
                    if on_transcript_chunk:
                        await on_transcript_chunk(text)
                    yield {
                        "type": "transcript_chunk",
                        "text": text,
                        "final": False,
                    }

                elif item["type"] == "done":
                    full_transcript = " ".join(transcript_parts).strip()
                    yield {
                        "type": "transcript_complete",
                        "transcript": full_transcript,
                        "final": True,
                    }
                    break

                elif item["type"] == "error":
                    yield {
                        "type": "error",
                        "error": item.get("error", "Unknown error"),
                        "final": True,
                    }
                    break

            await asyncio.gather(send_task, collect_task, return_exceptions=True)

        except Exception as e:
            logger.error(f"transcribe_stream failed: {e}", exc_info=True)
            yield {"type": "error", "error": str(e), "final": True}