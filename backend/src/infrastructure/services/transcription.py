"""
Transcription service.

Sits on top of SonicModel and provides a clean domain interface
for the agent to use on the file upload path.

The real-time WebSocket path does NOT go through this service —
it talks to SonicModel directly from the WebSocket route.
"""
import asyncio
from typing import Any, AsyncGenerator, Callable, Optional

from src.infrastructure.cache.service import CacheService
from src.infrastructure.language_models.sonic import SonicModel
from utils.logger import get_logger

logger = get_logger()

# Chunk size: 30 seconds of PCM16 audio at 16kHz mono
# PCM16 = 2 bytes/sample, 16000 samples/sec → 32000 bytes/sec
CHUNK_DURATION_SECONDS = 30
BYTES_PER_SECOND = 32_000  # 16kHz * 2 bytes (PCM16, mono)
CHUNK_SIZE_BYTES = CHUNK_DURATION_SECONDS * BYTES_PER_SECOND  # 960_000 bytes


TRANSCRIPT_TTL = 7200  # 2 hours
TRANSCRIPT_KEY_PREFIX = "transcript:realtime"

MEDICAL_SYSTEM_PROMPT = (
    "You are transcribing a recorded medical consultation between a doctor and patient. "
    "Transcribe all speech accurately and completely. "
    "Do not summarize, interpret, or add any commentary. "
    "Output only the spoken words."
)

# Demo transcript fallback when Sonic is unavailable in dev/test
DEMO_TRANSCRIPT = (
    "Doctor: Good morning Mr. Smith. How have you been since our last visit? "
    "Patient: Not too bad doctor, but my blood pressure has been a bit high lately. "
    "I've also been taking a potassium supplement I bought at the pharmacy. "
    "Doctor: I see. Your BP today is 148 over 90, which is higher than we'd like. "
    "You're currently on lisinopril 10mg. I'm a bit concerned about the potassium "
    "supplement combined with your lisinopril as that combination can cause high "
    "potassium levels in your blood. Please stop that supplement immediately. "
    "Patient: Oh I didn't know that. Should I be worried? "
    "Doctor: We'll monitor your levels. Your HbA1c came back at 7.4 percent which "
    "is slightly above our target. I'm going to increase your lisinopril to 20mg "
    "daily and I want to see you back in two weeks. Keep monitoring your BP at home "
    "and continue your metformin and aspirin as prescribed."
)


class TranscriptionService:
    """
    Transcription service for the file upload (one-shot) path.

    Wraps SonicModel.transcribe_bytes() with domain-specific defaults
    and error handling appropriate for medical consultations.

    Used by:
        ScribeAgent.process_consultation() when receiving a full audio file.

    NOT used by:
        The real-time WebSocket route — that calls SonicModel directly.
    """

    def __init__(
        self, 
        sonic_model,
        cache_service: Optional[CacheService] = None,
        max_concurrent_chunks: int = 5,
        chunk_size_bytes: int = CHUNK_SIZE_BYTES,
    ):
        """
        Args:
            sonic_model:             SonicModel instance (injected via DI container)
            cache_service:           Optional cache layer
            max_concurrent_chunks:   Max parallel Sonic API calls (tune to your rate limit)
            chunk_size_bytes:        Size of each audio chunk in bytes
        """
        self.sonic_model = sonic_model
        self.cache = cache_service
        self.chunk_size_bytes = chunk_size_bytes
        self._semaphore = asyncio.Semaphore(max_concurrent_chunks)

    def _realtime_key(self, patient_id: str) -> str:
        return f"{TRANSCRIPT_KEY_PREFIX}:{patient_id}"

    def _split_audio(self, audio_bytes: bytes) -> list[bytes]:
        """Split raw audio bytes into fixed-size chunks."""
        return [
            audio_bytes[i : i + self.chunk_size_bytes]
            for i in range(0, len(audio_bytes), self.chunk_size_bytes)
        ]

    async def _transcribe_chunk(
        self,
        chunk: bytes,
        index: int,
        system_prompt: str,
    ) -> tuple[int, str]:
        """
        Transcribe a single chunk under the concurrency semaphore.

        Returns:
            (index, transcript) so callers can reorder results.
        """
        async with self._semaphore:
            logger.debug(f"Transcribing chunk {index}: {len(chunk)} bytes")
            result = await self.sonic_model.transcribe_bytes(
                audio_bytes=chunk,
                system_prompt=system_prompt,
            )

        if not result.get("success"):
            logger.error(f"Chunk {index} failed: {result.get('error')}")
            return index, ""

        transcript = result.get("transcript", "")
        logger.debug(f"Chunk {index} done: {len(transcript)} chars")
        return index, transcript

    # ── File upload path ──────────────────────────────────────────────────────
    async def transcribe(
        self,
        audio_bytes: bytes,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Transcribe complete audio bytes to text.

        Args:
            audio_bytes: Raw PCM16 audio bytes from uploaded file
            system_prompt: Override the default medical transcription prompt

        Returns:
            Transcript string, or empty string on failure.
        """
        logger.info(f"TranscriptionService.transcribe: {len(audio_bytes)} bytes")

        # result = await self.sonic_model.transcribe_bytes(
        #     audio_bytes=audio_bytes,
        #     system_prompt=system_prompt or MEDICAL_SYSTEM_PROMPT,
        # )

        # if not result.get("success"):
        #     logger.error(f"Transcription failed: {result.get('error')}")
        #     return ""

        # transcript = result.get("transcript", "")
        # logger.info(f"Transcription complete: {len(transcript)} chars")
        # return transcript
        prompt = system_prompt or MEDICAL_SYSTEM_PROMPT

        chunks = self._split_audio(audio_bytes)
        logger.info(f"Split into {len(chunks)} chunk(s) of ~{self.chunk_size_bytes} bytes")

        if len(chunks) == 1:
            # Fast path: no chunking overhead for small files
            _, transcript = await self._transcribe_chunk(chunks[0], 0, prompt)
            return transcript

        # Dispatch all chunks concurrently (semaphore limits in-flight calls)
        tasks = [
            self._transcribe_chunk(chunk, i, prompt)
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Reassemble in order, skipping any failed chunks
        ordered: list[str] = []
        for result in sorted(
            (r for r in results if not isinstance(r, Exception)),
            key=lambda x: x[0],
        ):
            ordered.append(result[1])

        if any(isinstance(r, Exception) for r in results):
            logger.warning("Some chunks raised exceptions and were dropped")

        transcript = " ".join(filter(None, ordered))
        logger.info(f"Transcription complete: {len(transcript)} chars across {len(chunks)} chunks")
        return transcript

    # ── Real-time streaming path ──────────────────────────────────────────────

    async def process_real_time_audio(
        self,
        audio_bytes: bytes,
        patient_id: str,
        send_to_socket: Callable[[dict], Any],
    ) -> None:
        """
        Process a single real-time audio chunk from the WebSocket consumer.

        The browser sends one complete WebM/Opus blob every CHUNK_DURATION_MS.
        We transcribe it as a self-contained file via transcribe_bytes(), then:
          - Append the fragment to Redis for later full-transcript retrieval
          - Forward the fragment to the WebSocket client immediately

        Args:
            audio_bytes: Complete WebM/Opus blob for this time window
            patient_id: Patient identifier — used as Redis key
            send_to_socket: Async callable that sends a dict to the WebSocket client
        """
        cache_key = self._realtime_key(patient_id)

        try:
            logger.info(
                f"TranscriptionService: transcribing chunk "
                f"({len(audio_bytes)} bytes) for patient {patient_id}"
            )

            # Each browser chunk is a complete WebM blob — use transcribe_bytes,
            # NOT transcribe_stream. transcribe_bytes opens a fresh Sonic session,
            # decodes the audio, transcribes it, and returns cleanly.
            result = await self.sonic_model.transcribe_bytes(
                audio_bytes=audio_bytes,
                system_prompt=MEDICAL_SYSTEM_PROMPT,
            )

            if not result.get("success"):
                error_msg = result.get("error", "Transcription failed")
                logger.error(
                    f"Sonic chunk transcription failed for patient {patient_id}: {error_msg}"
                )

                # await send_to_socket({
                #     "event": "scribe.error",
                #     "error": error_msg,
                #     "patient_id": patient_id,
                # })
                # return
                pass

            fragment = result.get("transcript", "").strip()

            if not fragment:
                logger.info(
                    f"Empty transcript for patient {patient_id} chunk — "
                    "silence or no speech detected"
                )
                return

            logger.info(
                f"Chunk transcript for patient {patient_id}: "
                f"{len(fragment)} chars — '{fragment[:60]}...'"
            )

            # Accumulate in Redis for end_real_time_session retrieval
            await self._append_transcript(cache_key, fragment)

            # Forward to WebSocket client for live display
            await send_to_socket({
                "event": "scribe.transcript_chunk",
                "chunk": fragment,
                "patient_id": patient_id,
                "final": False,
            })

        except Exception as e:
            logger.error(f"process_real_time_audio failed: {e}", exc_info=True)
            await send_to_socket({
                "event": "scribe.error",
                "error": str(e),
                "patient_id": patient_id,
            })


    async def _append_transcript(self, cache_key: str, fragment: str) -> None:
        """
        Append a transcript fragment to the Redis cache.
        Creates a new entry if the key doesn't exist.
        Appends with a space separator if it does.
        """
        if not self.cache:
            return
        try:
            existing = await self.cache.get(cache_key)
            if existing:
                updated = existing.strip() + " " + fragment.strip()
            else:
                updated = fragment.strip()
            await self.cache.set(cache_key, updated, ttl=TRANSCRIPT_TTL)
        except Exception as e:
            logger.error(f"_append_transcript failed: {e}", exc_info=True)

    async def get_accumulated_transcript(self, patient_id: str) -> str:
        """
        Retrieve the full accumulated real-time transcript for a patient.
        Called by the consumer when the 'end' event is received.

        Args:
            patient_id: Patient identifier

        Returns:
            Full transcript string, or empty string if nothing accumulated
        """
        if not self.cache:
            return ""
        try:
            transcript = await self.cache.get(self._realtime_key(patient_id))
            return transcript or ""
        except Exception as e:
            logger.error(f"get_accumulated_transcript failed: {e}", exc_info=True)
            return ""

    async def clear_realtime_transcript(self, patient_id: str) -> None:
        """Clear the accumulated real-time transcript for a patient after session ends."""
        if not self.cache:
            return
        try:
            await self.cache.invalidate(self._realtime_key(patient_id))
        except Exception as e:
            logger.error(f"clear_realtime_transcript failed: {e}", exc_info=True)