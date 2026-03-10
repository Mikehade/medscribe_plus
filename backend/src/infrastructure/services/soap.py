"""
SOAP service.

Handles SOAP note generation and session transcript management.
All LLM calls go through BedrockModel. Cache-backed transcript storage.

Called by SOAPTools — no direct agent or tool logic here.
"""
import json
from typing import Any, Dict, Optional

from src.infrastructure.cache.service import CacheService
from utils.logger import get_logger

logger = get_logger()

TRANSCRIPT_TTL = 7200   # 2 hours
SOAP_TTL = 7200

SOAP_SYSTEM_PROMPT = """
You are a clinical documentation assistant. Given a doctor-patient consultation transcript,
generate a structured SOAP note.

Respond ONLY with valid JSON matching this exact schema — no preamble, no markdown fences:

{
  "subjective": "Patient's chief complaint, symptoms, history of present illness",
  "objective": "Vitals, physical exam findings, lab results if mentioned",
  "assessment": "Diagnosis and clinical reasoning",
  "plan": "Treatment plan, medications prescribed, referrals",
  "icd10_codes": ["code1", "code2"],
  "cpt_codes": ["code1"],
  "medications_mentioned": ["drug name dose frequency"],
  "follow_up": "Follow-up instructions and timeline",
  "conditions_detected": ["condition1", "condition2"]
}

Rules:
- Never invent symptoms, vitals, or medications not in the transcript
- If a section has no data, use an empty string — never fabricate
- conditions_detected should list the primary medical conditions discussed
"""


class SOAPService:
    """
    SOAP note generation and session transcript management.

    Responsibilities:
    - Generate structured SOAP notes from transcripts via Nova 2 Lite
    - Store and retrieve session transcripts (cache-backed)
    - Append transcript chunks for real-time sessions

    All LLM calls go through the injected BedrockModel instance.
    """

    def __init__(self, llm_model: Any, cache_service: CacheService):
        self.llm_model = llm_model
        self.cache = cache_service

    def _transcript_key(self, session_id: str) -> str:
        return f"transcript:{session_id}"

    def _soap_key(self, session_id: str) -> str:
        return f"soap:{session_id}"

    async def generate_soap_note(
        self,
        transcript: str,
        patient_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured SOAP note from a consultation transcript.

        Args:
            transcript: Full consultation transcript text
            patient_context: Optional patient history for additional context
            session_id: Optional session ID to cache the result

        Returns:
            {"success": True, "data": {soap dict}} or {"success": False, "error": "..."}
        """
        try:
            # Check cache first
            if session_id:
                cache_key = self._soap_key(session_id)
                cached = await self.cache.get(cache_key)
                if cached:
                    logger.debug(f"SOAP cache HIT: {session_id}")
                    return {"success": True, "data": cached}

            # Build user message — include patient context if available
            user_message = f"CONSULTATION TRANSCRIPT:\n{transcript}"
            if patient_context:
                context_str = json.dumps(patient_context, indent=2)
                user_message = (
                    f"PATIENT CONTEXT:\n{context_str}\n\n"
                    f"CONSULTATION TRANSCRIPT:\n{transcript}"
                )

            # Call Nova 2 Lite via BedrockModel
            response = None
            prompt_output = self.llm_model.prompt(
                text=user_message,
                system_prompt=SOAP_SYSTEM_PROMPT,
                stream=False,
                enable_tools=False,
                temperature=0.1,    # low temp for deterministic clinical output
                max_tokens=2048,
            )

            generator = None
            import asyncio
            if hasattr(prompt_output, "__aiter__"):
                async def _collect():
                    result = None
                    async for chunk in prompt_output:
                        result = chunk
                    return result
                response = await _collect()
            elif asyncio.iscoroutine(prompt_output):
                response = await prompt_output
            else:
                response = prompt_output

            if not response:
                return {"success": False, "error": "No response from LLM"}

            # Extract text and parse JSON
            raw_text = self.llm_model.extract_text_response(response)
            clean = raw_text.strip().strip("```json").strip("```").strip()
            soap_data = json.loads(clean)

            # Cache result
            if session_id:
                await self.cache.set(self._soap_key(session_id), soap_data, ttl=SOAP_TTL)

            logger.info(f"SOAP note generated for session {session_id}")
            return {"success": True, "data": soap_data}

        except json.JSONDecodeError as e:
            logger.error(f"SOAP JSON parse failed: {e} — raw: {raw_text[:200]}")
            return {"success": False, "error": f"Failed to parse SOAP JSON: {e}"}
        except Exception as e:
            logger.error(f"generate_soap_note failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def save_transcript_chunk(
        self,
        session_id: str,
        chunk: str,
    ) -> Dict[str, Any]:
        """
        Append a transcript chunk to the session transcript in cache.
        Creates new entry if none exists, appends if one does.

        Args:
            session_id: Session identifier
            chunk: Transcript text to append

        Returns:
            {"success": True, "full_transcript": "..."}
        """
        try:
            cache_key = self._transcript_key(session_id)
            existing = await self.cache.get(cache_key) or ""
            updated = (existing + " " + chunk).strip() if existing else chunk
            await self.cache.set(cache_key, updated, ttl=TRANSCRIPT_TTL)
            return {"success": True, "full_transcript": updated}

        except Exception as e:
            logger.error(f"save_transcript_chunk failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_session_transcript(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve the full accumulated transcript for a session.

        Args:
            session_id: Session identifier

        Returns:
            {"success": True, "transcript": "..."}
        """
        try:
            cache_key = self._transcript_key(session_id)
            transcript = await self.cache.get(cache_key) or ""
            return {"success": True, "transcript": transcript}

        except Exception as e:
            logger.error(f"get_session_transcript failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def clear_session(self, session_id: str) -> None:
        """Clear transcript and SOAP cache for a session."""
        await self.cache.invalidate(self._transcript_key(session_id))
        await self.cache.invalidate(self._soap_key(session_id))