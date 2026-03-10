"""
Patient service.

All patient and EHR operations. Cache-backed (no DB for now).
Patient ID is the primary identifier for all operations.

Called by PatientTools — no direct agent or tool logic here.
"""
import json
from typing import Any, Dict, List, Optional

from src.infrastructure.cache.service import CacheService
from utils.logger import get_logger

logger = get_logger()

# Cache TTLs
PATIENT_CONTEXT_TTL = 3600  # 1 hour

# Mock EHR data — replace with real EHR integration later
MOCK_EHR_DB = {
    "P001": {
        "name": "John Smith",
        "dob": "1965-03-15",
        "allergies": ["penicillin", "sulfa drugs"],
        "medications": [
            {"name": "lisinopril", "dose": "10mg", "frequency": "daily"},
            {"name": "metformin", "dose": "500mg", "frequency": "twice daily"},
            {"name": "aspirin", "dose": "81mg", "frequency": "daily"},
        ],
        "conditions": ["hypertension", "type 2 diabetes"],
        "last_visit": "2024-11-12",
        "prior_notes": [
            {
                "date": "2024-11-12",
                "summary": "Routine follow-up. BP 138/85. HbA1c 7.2%. Medications continued.",
            }
        ],
    }
}


class PatientService:
    """
    Patient and EHR operations service.

    Responsibilities:
    - Retrieve patient history and context
    - Insert finalized SOAP notes into EHR
    - Flag missing required EHR fields
    - Cache patient context by patient_id

    All data is cache-backed. No DB/SQLAlchemy.
    """

    def __init__(self, cache_service: CacheService):
        self.cache = cache_service

    def _patient_key(self, patient_id: str) -> str:
        return f"patient:context:{patient_id}"

    def _ehr_notes_key(self, patient_id: str) -> str:
        return f"patient:ehr_notes:{patient_id}"

    async def get_patient_history(self, patient_id: str) -> Dict[str, Any]:
        """
        Retrieve patient history, medications, allergies, conditions.
        Checks cache first, falls back to mock EHR.

        Args:
            patient_id: Patient identifier

        Returns:
            {"success": True, "data": {...patient context...}}
        """
        try:
            cache_key = self._patient_key(patient_id)
            cached = await self.cache.get(cache_key)
            if cached:
                logger.debug(f"Patient context cache HIT: {patient_id}")
                return {"success": True, "data": cached}

            # Fall back to mock EHR
            patient = MOCK_EHR_DB.get(patient_id) or MOCK_EHR_DB.get("P001")
            if not patient:
                return {"success": False, "error": f"Patient {patient_id} not found"}

            await self.cache.set(cache_key, patient, ttl=PATIENT_CONTEXT_TTL)
            logger.info(f"Patient context loaded: {patient_id}")
            return {"success": True, "data": patient}

        except Exception as e:
            logger.error(f"get_patient_history failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def insert_ehr_note(
        self,
        patient_id: str,
        soap_note: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Insert finalized SOAP note into EHR (cached).
        In production this would call Nova Act to navigate the real EHR.

        Args:
            patient_id: Patient identifier
            soap_note: Finalized SOAP note dict
            session_id: Session identifier for audit trail

        Returns:
            {"success": True, "ehr_record_id": "..."}
        """
        try:
            import uuid as _uuid
            record_id = f"NOTE-{session_id or _uuid.uuid4()}"
            cache_key = self._ehr_notes_key(patient_id)

            # Append to existing notes list
            existing = await self.cache.get(cache_key) or []
            existing.append({
                "record_id": record_id,
                "soap_note": soap_note,
                "session_id": session_id,
                "inserted_at": _uuid.uuid4().hex,  # placeholder timestamp
            })
            await self.cache.set(cache_key, existing, ttl=PATIENT_CONTEXT_TTL * 24)

            logger.info(f"EHR note inserted: {record_id} for patient {patient_id}")
            return {
                "success": True,
                "ehr_record_id": record_id,
                "message": f"Note inserted into EHR for patient {patient_id}",
            }

        except Exception as e:
            logger.error(f"insert_ehr_note failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def flag_missing_ehr_fields(
        self,
        patient_id: str,
        soap_note: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check SOAP note for required EHR fields that are missing or empty.

        Args:
            patient_id: Patient identifier
            soap_note: SOAP note dict to validate

        Returns:
            {"success": True, "missing_fields": [...], "is_complete": bool}
        """
        try:
            required = {
                "subjective": "Chief complaint / subjective findings",
                "objective": "Objective findings / vitals",
                "assessment": "Clinical assessment / diagnosis",
                "plan": "Treatment plan",
                "follow_up": "Follow-up instructions",
            }

            missing = [
                {"field": field, "label": label}
                for field, label in required.items()
                if not str(soap_note.get(field, "")).strip()
            ]

            return {
                "success": True,
                "missing_fields": missing,
                "is_complete": len(missing) == 0,
            }

        except Exception as e:
            logger.error(f"flag_missing_ehr_fields failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}