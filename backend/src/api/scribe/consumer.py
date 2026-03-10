"""
Scribe WebSocket consumer.

Handles real-time audio streaming and consultation processing.

Events (incoming from client):
    transcribe      — audio chunk bytes, keeps transcribing in real time
    end             — session over, process full accumulated transcript
    pong            — heartbeat response, no-op

Events (outgoing to client):
    scribe.transcript_chunk     — live transcript fragment as it arrives
    scribe.consultation_result  — full result after 'end' (soap, scores, etc.)
    scribe.processing           — status update while agent runs
    scribe.error                — error occurred
"""
import asyncio
import json
import base64
import uuid
from typing import Any, Dict, Optional

from fastapi import WebSocket

from src.api.base.consumer import BaseWebSocketConsumer
from src.core.agents.scribe import ScribeAgent
# from src.infrastructure.db.models.auth import AuthProfile
from utils.logger import get_logger

logger = get_logger()


class ScribeConsumer(BaseWebSocketConsumer):
    """
    WebSocket consumer for MedScribe+ real-time audio transcription.

    Lifecycle:
    1. Client connects → on_connect fires
    2. Client sends 'transcribe' events with base64 audio chunks
       → each chunk goes to ScribeAgent.process_real_time_audio()
       → transcript fragments cached in Redis, forwarded to client
    3. Client sends 'end' event
       → ScribeAgent.end_real_time_session() retrieves full transcript
       → Runs full pipeline (SOAP + evaluation)
       → Result sent back as 'scribe.consultation_result'

    Expected message format from client:
        {"event": "transcribe", "audio": "<base64 encoded bytes>", "patient_id": "P001"}
        {"event": "end", "patient_id": "P001", "session_id": "..."}
        {"message": {"text": "pong"}}
    """

    def __init__(
        self,
        websocket: WebSocket,
        # user: AuthProfile,
        agent: ScribeAgent,
        user: Any = None,
    ):
        super().__init__(websocket, user)
        self.agent = agent
        self.user = user
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.connected = False
        self.current_session_id: Optional[str] = None
        self.current_patient_id: Optional[str] = None

        # Event name constants
        self.events = {
            "transcript_chunk": "scribe.transcript_chunk",
            "consultation_result": "scribe.consultation_result",
            "processing": "scribe.processing",
            "error": "scribe.error",
            "connected": "scribe.connected",
        }

    async def on_connect(self) -> None:
        """Called after WebSocket handshake completes."""
        self.connected = True
        self.current_session_id = str(uuid.uuid4())
        logger.info(
            # f"ScribeConsumer connected: user={self.user.user_id} "
            f"session={self.current_session_id}"
        )

        await self._send_message({
            "event": self.events["connected"],
            "session_id": self.current_session_id,
            "message": "Connected to MedScribe+. Ready for audio.",
        })

        self.heartbeat_task = asyncio.create_task(self.heartbeat())

    async def on_disconnect(self) -> None:
        """Called before WebSocket closes."""
        self.connected = False

        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass

        logger.info(
            # f"ScribeConsumer disconnected: user={self.user.user_id} "
            f"session={self.current_session_id}"
        )

    async def handle_message(self, message: Dict[str, Any]) -> None:
        """
        Route incoming WebSocket messages to the correct handler.

        Args:
            message: Parsed JSON message dict from client
        """
        event = message.get("event")

        if event == "transcribe":
            await self._handle_transcribe(message)

        elif event == "end":
            await self._handle_end(message)

        elif message.get("message", {}).get("text") == "pong":
            pass  # heartbeat pong — no action needed

        else:
            logger.warning(f"ScribeConsumer: unknown event '{event}'")
            await self.send_error(f"Unknown event: {event}", 400)

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _handle_transcribe(self, message: Dict[str, Any]) -> None:
        """
        Handle a real-time audio chunk from the browser microphone.

        Expected message:
            {
                "event": "transcribe",
                "audio": "<base64 encoded PCM16 bytes>",
                "patient_id": "P001",       # optional, defaults to P001
                "session_id": "..."          # optional, uses current session
            }
        """
        try:
            audio_b64 = message.get("audio", "")
            if not audio_b64:
                await self.send_error("Missing audio data in transcribe event", 400)
                return

            patient_id = message.get("patient_id", "P001")
            session_id = message.get("session_id") or self.current_session_id

            # Store patient_id for session
            self.current_patient_id = patient_id

            # Decode base64 audio from browser
            try:
                audio_bytes = base64.b64decode(audio_b64)
            except Exception:
                await self.send_error("Invalid base64 audio data", 400)
                return

            # Process chunk — appends to Redis, sends transcript back through socket
            await self.agent.process_real_time_audio(
                audio_bytes=audio_bytes,
                patient_id=patient_id,
                session_id=session_id,
                send_to_socket=self._send_message,
            )

        except Exception as e:
            logger.error(f"_handle_transcribe failed: {e}", exc_info=True)
            await self.send_error("Error processing audio chunk", 500)

    async def _handle_end(self, message: Dict[str, Any]) -> None:
        """
        Handle session end — retrieve full transcript and run consultation pipeline.

        Expected message:
            {
                "event": "end",
                "patient_id": "P001",
                "session_id": "..."    # optional, uses current session
            }
        """
        try:
            patient_id = message.get("patient_id") or self.current_patient_id or "P001"
            session_id = message.get("session_id") or self.current_session_id

            logger.info(
                f"ScribeConsumer: session end received "
                f"patient={patient_id} session={session_id}"
            )

            # Notify client that processing has started
            await self._send_message({
                "event": self.events["processing"],
                "message": "Consultation ended. Generating SOAP note and evaluation...",
                "session_id": session_id,
            })

            # Run the full pipeline — same as file upload path
            result = await self.agent.end_real_time_session(
                patient_id=patient_id,
                session_id=session_id,
            )

            # Send result back to client
            if result.get("success"):
                await self._send_message({
                    "event": self.events["consultation_result"],
                    "session_id": session_id,
                    "patient_id": patient_id,
                    "data": {
                        "soap": result.get("soap", {}),
                        "scores": result.get("scores", {}),
                        "patient_context": result.get("patient_context", {}),
                        "missing_fields": result.get("missing_fields", []),
                        "transcript": result.get("transcript", ""),
                    },
                })
                logger.info(f"Consultation result sent for session={session_id}")
            else:
                await self.send_error(
                    result.get("error", "Consultation processing failed"), 500
                )

            # Reset session for potential reuse
            self.current_session_id = str(uuid.uuid4())
            self.current_patient_id = None

        except Exception as e:
            logger.error(f"_handle_end failed: {e}", exc_info=True)
            await self.send_error("Error processing consultation", 500)