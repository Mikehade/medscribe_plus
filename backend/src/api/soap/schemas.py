from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class SoapRequest(BaseModel):
    session_id: str

class SoapResponse(BaseModel):
    status: str
    data: Dict[str, Any]

