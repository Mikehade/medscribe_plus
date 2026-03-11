"""
ScribeAgent — orchestrates the MedScribe+ pipeline.

Option A (live audio): Strands BidiAgent handles Sonic streaming directly
                       via the WebSocket consumer. This agent is NOT used
                       for Option A's Sonic session.

Option B (file upload): This agent receives a transcript and orchestrates
                        SOAP generation + evaluation + EHR insertion.
"""
import json
import uuid
import asyncio
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from src.core.agents.base import BaseAgent
from src.core.tools.base import ToolRegistry
from src.core.prompts.scribe import ScribePrompt

from src.infrastructure.services.transcription import TranscriptionService
from src.infrastructure.services.soap import SOAPService
from src.infrastructure.services.evaluation import EvaluationService
from src.infrastructure.cache.service import CacheService
from utils.logger import get_logger

logger = get_logger()


class ScribeAgent(BaseAgent):
    """
    Orchestrates the full MedScribe+ pipeline for file-based consultations.

    Flow:
        transcript → SOAP note → evaluation (parallel) → dashboard scores → EHR insert
    """

    def __init__(
        self,
        llm_model: Any,                      # BedrockModel instance
        tool_registry: ToolRegistry,
        prompt_template: ScribePrompt,
        transcription_service: TranscriptionService,
        soap_service: SOAPService,
        evaluation_service: EvaluationService,
        cache_service: CacheService,
    ):
        super().__init__()
        self.llm_model = llm_model
        self.tool_registry = tool_registry
        self.prompt_template = prompt_template
        self.transcription_service = transcription_service
        self.soap_service = soap_service
        self.evaluation_service = evaluation_service
        self.cache = cache_service

        # Set tool registry on the LLM model
        self.llm_model.tool_registry = self.tool_registry

        # Log tool setup
        if self.tool_registry:
            available_tools = self.tool_registry.get_available_tools()
            logger.info(f"Agent initialized with {len(available_tools)} tools: {available_tools}")
        else:
            logger.warning("Agent initialized without tool registry")

    async def process_message(
        self,
        user=None,
        message: str = "",
        bot: str = "scribe",
        use_history: bool = False,
        enable_tools: bool = True,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Not used directly — use process_consultation instead."""
        return await self.process_consultation(transcript=message)

    async def process_consultation(
        self,
        transcript: str,
        patient_id: str = "P001",
        session_id: Optional[str] = None,
        enable_tools: bool = True,
    ) -> Dict[str, Any]:
        """
        Full pipeline: transcript → SOAP → evaluate → scores.

        Args:
            transcript: Full consultation transcript text
            patient_id: Patient identifier for EHR lookup
            session_id: Optional session ID for audit trail

        Returns:
            {
                "soap": {...},
                "scores": {...},
                "patient_context": {...},
                "session_id": "..."
            }
        """
        session_id = session_id or str(uuid.uuid4())
        logger.info(f"Processing consultation session {session_id}")

        try:
            # Add patient_id context to tool registry for tool execution

            # TODO: - Valdate that trsncript is actually between doctor and patient, if otherwise, flag and return here

            # Tools read patient_id, session_id, transcript from kwargs
            # The LLM sets soap_note, patient_context, conditions as it generates them
            if enable_tools and self.tool_registry:
                for tool in self.tool_registry.tool_classes:
                    tool.kwargs.update({
                        "patient_id": patient_id,
                        "session_id": session_id,
                        "transcript": transcript,
                        "patient_context": None,   # populated after get_patient_history fires
                        "soap_note": {},            # populated after generate_soap_note fires
                        "conditions": [],           # populated from patient context
                    })
                
            # Prepare prompts
            system_prompt = self.prompt_template.get_system_prompt(
                current_date=datetime.now().strftime("%Y-%m-%d")
            )

            # ── Run the LLM loop ──────────────────────────────────────────────
            # The LLM drives tool calls. We run until the loop completes.
            prompt_output = self.llm_model.prompt(
                text=f"Process this consultation transcript:\n\n{transcript}",
                system_prompt=system_prompt,
                stream=False,
                enable_tools=True,
                # reasoning=True,
            )

            generator = await self.ensure_async_generator(prompt_output)
            async for _ in generator:
                pass  # Let the LLM loop run — tool calls fire automatically

            # ── Fetch results from services after the loop ────────────────────
            # The LLM called the tools which cached results — read them here
            soap_result = await self.soap_service.get_session_transcript(session_id)
            # Prefer the cached SOAP note over re-generating
            cached_soap = await self.cache.get(f"soap:{session_id}") or {}

            # Read cached evaluation scores
            cached_scores = await self.cache.get(f"evaluation:scores:{session_id}") or {}

            # Read patient context
            patient_context = await self.cache.get(f"patient:context:{patient_id}") or {}

            # Check for missing fields in the SOAP note
            # Not in soap service yet
            # missing_result = await self.soap_service.flag_missing_fields(
            #     session_id=session_id,
            #     soap_note=cached_soap,
            # ) if cached_soap else {"missing_fields": []}
            # missing_fields = missing_result.get("missing_fields", [])

            missing_fields = []

            logger.info(f"Consultation complete: session={session_id}")
            return {
                "success": True,
                "session_id": session_id,
                "soap": cached_soap,
                "scores": cached_scores,
                "patient_context": patient_context,
                "missing_fields": missing_fields,
                "transcript": transcript,
            }


        except Exception as e:
            logger.error(f"Consultation processing failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "session_id": session_id}

    async def approve_and_insert(
        self,
        session_id: str,
        soap_data: Dict[str, Any],
        patient_id: str = "P001",
    ) -> Dict[str, Any]:
        """
        Physician approval step — insert finalized note into EHR.
        Called after physician reviews and approves the SOAP note.

        Args:
            session_id: Session identifier for audit trail
            soap_data: Finalized SOAP note dict
            patient_id: Patient identifier

        Returns:
            {"success": True, "ehr_record_id": "..."}
        """
        from src.infrastructure.services.patient import PatientService
        # PatientService is called directly here — not via tools
        # because this is a post-LLM-loop explicit action
        cached_patient = await self.cache.get(f"patient:context:{patient_id}")
        if not cached_patient:
            logger.warning(f"No patient context in cache for {patient_id} during EHR insert")

        # Delegate through the tool so it goes via the service layer
        patient_tools = next(
            (t for t in self.tool_registry.tool_classes
             if hasattr(t, '_insert_ehr_note')),
            None
        )
        if patient_tools:
            patient_tools.kwargs.update({
                "patient_id": patient_id,
                "session_id": session_id,
            })
            return await patient_tools._insert_ehr_note(soap_note=soap_data)

        logger.error("PatientTools not found in registry for EHR insert")
        return {"success": False, "error": "PatientTools not registered"}

    async def process_audio(
        self,
        audio: bytes,
        patient_id: str = "P001",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process Audio.

        Args:
            audio: Uploaded audio bytes
            patient_id: Patient identifier for EHR lookup
            session_id: Optional session ID for audit trail

        Returns:
            {
                "soap": {...},
                "scores": {...},
                "patient_context": {...},
                "session_id": "..."
            }
        """
        session_id = session_id or str(uuid.uuid4())
        logger.info(f"Processing audio for session {session_id}")

        try:
            audio = await audio.read()
            logger.info(f"\n In processing audio: {type(audio)}  \n")

            # transcribe audio
            # this already works but disabled for faster testing, I commented it out
            # transcript = await self.transcription_service.transcribe(audio)

            transcript = (
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

            logger.info(f"\n Transcript in processing audio: {transcript}  \n")

            # return {
            #     "success": False,
            #     "session_id": session_id,
            #     # "soap": soap_data,
            #     # "scores": scores_result.get("scores", {}),
            #     # "patient_context": patient_context,
            #     # "missing_fields": missing_result.get("missing_fields", []),
            # }
            logger.info(f"Transcription complete: {len(transcript)} chars")

            return await self.process_consultation(
                transcript=transcript,
                patient_id=patient_id,
                session_id=session_id,
            )

        except Exception as e:
            logger.error(f"Consultation processing failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "session_id": session_id}

    async def process_real_time_audio(
        self,
        audio_bytes: bytes,
        patient_id: str,
        session_id: str,
        send_to_socket: Callable[[dict], Any],
    ) -> None:
        """
        Handle a single real-time audio chunk from the WebSocket consumer.

        - Passes bytes to TranscriptionService.process_real_time_audio()
        - TranscriptionService appends transcript to Redis and sends chunks to socket
        - Does NOT trigger SOAP generation — that happens on 'end' event

        Args:
            audio_bytes: Raw audio chunk bytes from the browser mic
            patient_id: Patient identifier for Redis keying
            session_id: Session identifier
            send_to_socket: Async callable to push events to the WebSocket client
        """
        await self.transcription_service.process_real_time_audio(
            audio_bytes=audio_bytes,
            patient_id=patient_id,
            send_to_socket=send_to_socket,
        )

    async def end_real_time_session(
        self,
        patient_id: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Called when the consumer receives the 'end' event.

        Retrieves the full accumulated transcript from Redis and runs
        process_consultation — the same pipeline as the file upload path.

        Args:
            patient_id: Patient identifier
            session_id: Session identifier

        Returns:
            Full consultation result dict (soap, scores, etc.)
        """
        logger.info(f"Ending real-time session: patient={patient_id} session={session_id}")

        # Get full accumulated transcript from Redis
        transcript = await self.transcription_service.get_accumulated_transcript(patient_id)

        if not transcript:
            logger.warning(f"No transcript found for patient {patient_id} — using demo")
            from src.infrastructure.services.transcription import DEMO_TRANSCRIPT
            transcript = DEMO_TRANSCRIPT

        # Run the same pipeline as file upload
        result = await self.process_consultation(
            transcript=transcript,
            patient_id=patient_id,
            session_id=session_id,
        )

        # Clean up the real-time transcript cache after processing
        await self.transcription_service.clear_realtime_transcript(patient_id)

        return result