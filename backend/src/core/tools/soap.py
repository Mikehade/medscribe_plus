"""
SOAP note generation and extraction tools.
"""
from typing import Any, Dict, List, Optional
from src.core.tools.base import BaseTool

from src.infrastructure.services.soap import SOAPService
from utils.logger import get_logger

logger = get_logger()


class SOAPTools(BaseTool):
    """
    Tools for SOAP note generation, ICD-10 extraction, and CPT code suggestion.
    """

    def __init__(
        self,
        soap_service: SOAPService,
        enabled_tools: list = None,
        **kwargs
    ) -> None:
        super().__init__(enabled_tools=enabled_tools, **kwargs)
        self.soap_service = soap_service

    async def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a currency tool."""
        method = self.get_tool_method(tool_name)
        
        if not method:
            return {
                "success": False,
                "error": f"Tool method '{tool_name}' not found"
            }
        
        try:
            tool_input_with_context = {**tool_input, **self.kwargs}
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

    async def _generate_soap_note(
        self,
        # transcript: str,
        patient_context: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured SOAP note from consultation transcript."""
        try:

            # return {"success": True, "data": soap_data, "message": "SOAP note generated"}
            session_id = kwargs.get("session_id")
            transcript = kwargs.get("transcript", "")
            # patient_context = self.kwargs.get("patient_context")
            return await self.soap_service.generate_soap_note(
                transcript=transcript,
                patient_context=patient_context,
                session_id=session_id,
            )

        except Exception as e:
            logger.error(f"SOAP generation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # async def _save_session_transcript(
    #     self,
    #     **kwargs
    # ) -> Dict[str, Any]:
    #     """Append a transcript chunk to the session store."""
    #     try:
    #         # In production: write to Redis/DynamoDB keyed by session_id
    #         # For hackathon: in-memory via cache service
    #         cache = kwargs.get("cache_service")
    #         if cache:
    #             existing = await cache.get(f"transcript:{session_id}") or ""
    #             updated = existing + " " + transcript_chunk
    #             await cache.set(f"transcript:{session_id}", updated, ttl=3600)
    #         return {"success": True, "message": "Transcript chunk saved"}
    #     except Exception as e:
    #         return {"success": False, "error": str(e)}

    async def _get_session_transcript(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        """Retrieve full transcript for a session."""
        try:
            
            # return {"success": False, "error": "No cache service"}
            session_id = kwargs.get("session_id", "")
            return await self.soap_service.get_session_transcript(session_id=session_id)

        except Exception as e:
            return {"success": False, "error": str(e)}

    # fmt: on