
"""
Request/response schemas for MedScribe API.
"""
from uuid import UUID
from typing import Optional, Dict, Union, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class AudioUploadResponse(BaseModel):
    session_id: str
    transcript: str
    soap: Dict[str, Any]
    scores: Dict[str, Any]
    patient_context: Dict[str, Any]
    missing_fields: List[Dict[str, str]]


class ApproveRequest(BaseModel):
    session_id: str
    soap: Dict[str, Any]
    patient_id: str = "P001"


class ApproveResponse(BaseModel):
    success: bool
    ehr_record_id: Optional[str] = None
    message: str


class TranscriptChunkRequest(BaseModel):
    session_id: str
    chunk: str


class SessionScoresResponse(BaseModel):
    session_id: str
    scores: Dict[str, Any]

class PatientContextResponse(BaseModel):
    success: bool
    data: Dict[str, Any]

class PatientContextRequest(BaseModel):
    patient_id: str