"""
Patient tools for LLM agents.
"""
from typing import Any, Dict, Optional
from src.core.tools.base import BaseTool

from src.infrastructure.services.patient import PatientService
from utils.logger import get_logger

logger = get_logger()

class PatientTools(BaseTool):
    """
    Tools for patients operations.

    Args:
        - patient_service: - 
        - enabled_tools: - list of tools to be enabled for an agent using this tool
    """
    
    def __init__(
        self, 
        patient_service: PatientService, 
        enabled_tools: list = None, 
        **kwargs
    ) -> None:
        super().__init__(enabled_tools=enabled_tools, **kwargs)
        self.patient_service = patient_service
    
    async def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a currency tool."""
        method = self.get_tool_method(tool_name)
        
        if not method:
            return {
                "success": False,
                "error": f"Tool method '{tool_name}' not found"
            }
        
        try:
            # tool_input_with_context = {**tool_input, **self.kwargs}
            tool_input_with_context = {**self.kwargs, **tool_input}
            result = await method(**tool_input_with_context)
            
            # Ensure result is a dictionary
            if isinstance(result, tuple):
                if len(result) == 3:
                    success, data, message = result
                    return {
                        "success": success,
                        "data": data,
                        "message": message
                    }
            elif isinstance(result, dict):
                return result
            
            return {"success": True, "data": result}
            
        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to execute {tool_name}"
            }

    # fmt: off

    # All tools/functions should be defined in this range
    async def _get_patient_history(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        """Retrieve patient medical history, medications, allergies and prior notes from EHR."""
        try:
            
            
            patient_id = kwargs.get("patient_id", "P001")
            return await self.patient_service.get_patient_history(patient_id=patient_id)
        
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _insert_ehr_note(
        self,
        soap_note: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Insert finalized SOAP note into EHR via Nova Act browser automation."""
        try:
            

            patient_id = kwargs.get("patient_id", "P001")
            session_id = kwargs.get("session_id")
            return await self.patient_service.insert_ehr_note(
                patient_id=patient_id,
                soap_note=soap_note,
                session_id=session_id,
            )
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _flag_missing_ehr_fields(
        self,
        soap_note: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Identify required EHR fields missing from the SOAP note."""
        try:

             # Handle case where soap_note arrives as a JSON string
            if isinstance(soap_note, str):
                import json
                try:
                    soap_note = json.loads(soap_note)
                except json.JSONDecodeError:
                    logger.warning("soap_note is a string but not valid JSON, wrapping as raw text")
                    soap_note = {"subjective": soap_note}
            logger.info(f"[In Tool]: Flagging missing EHR fields for {soap_note}")
            patient_id = kwargs.get("patient_id", "P001")
            return await self.patient_service.flag_missing_ehr_fields(
                patient_id=patient_id,
                soap_note=soap_note,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    # fmt: on