"""
Scribe evaluation tools — bridge between ScribeAgent and EvaluationAgent.
 
Exposes a single tool to the ScribeAgent: _evaluate_consultation.
When the ScribeAgent LLM calls this tool, it triggers EvaluationAgent.process_message()
which runs its own LLM loop with EvaluationTools.
 
ScribeAgent never knows about individual evaluation checks.
Zero implementation here — routing only.
"""
import json
from typing import Any, Dict
 
from src.core.tools.base import BaseTool
from src.core.agents.evaluation import EvaluationAgent
from utils.logger import get_logger
 
logger = get_logger()


class ScribeEvaluationTools(BaseTool):
    """
    Evaluation tools for the ScribeAgent.

    Exposes a single tool to the ScribeAgent — evaluate_consultation.
    This triggers EvaluationService.run_full_evaluation() which runs
    all checks (hallucinations, drug interactions, guideline alignment,
    completeness) in parallel via its internal EvaluationAgent.

    kwargs injected by ScribeAgent before execution:
        session_id      — for caching scores
        transcript      — raw consultation transcript
        soap_note       — generated SOAP note dict
        conditions      — patient conditions list for guideline check
    """

    def __init__(
        self, 
        evaluation_agent: EvaluationAgent,
        enabled_tools: list[str] = None, 
        cache_service: Any = None,
        **kwargs
    ):
        super().__init__(enabled_tools=enabled_tools, **kwargs)
        self.evaluation_agent = evaluation_agent
        self.cache_service = cache_service

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

    async def _evaluate_consultation(self, **kwargs) -> Dict[str, Any]:
        """
        Run a full clinical evaluation of the generated SOAP note.
        Checks for hallucinations, drug interactions, clinical guideline alignment,
        and documentation completeness. Returns aggregated scores ready for the
        physician dashboard. Call this after the SOAP note has been generated.
        """
        # Build the context message for EvaluationAgent.process_message
        transcript = kwargs.get("transcript", "")
        session_id = kwargs.get("session_id")

        # Prefer the cached SOAP note over re-generating
        soap_note = await self.cache_service.get(f"soap:{session_id}") or {}
        logger.info(f"SOAP note: {soap_note}\n")
        conditions = soap_note.get("conditions_detected", [])

        logger.debug(
            f"evaluate_consultation: session={session_id} "
            f"soap_fields={list(soap_note.keys()) if isinstance(soap_note, dict) else 'str'} "
            f"conditions={conditions}"
            f"transcript={transcript}"
            # f"patient context={patient_context}"
        )

        if not soap_note and session_id and hasattr(self, "_cache"):
            soap_note = await self._cache.get(f"soap:{session_id}") or {}
            logger.debug(f"evaluate_consultation: loaded SOAP from cache for session={session_id}")
        
        # logger.info(f"evaluate_consultation kwargs: soap_note={kwargs}")
 
        context = json.dumps({
            "soap_note_json": json.dumps(soap_note) if isinstance(soap_note, dict) else soap_note,
            "transcript": transcript,
            "conditions": conditions,
            "session_id": session_id,
        })
 
        # EvaluationAgent.process_message runs the full LLM loop with its own tools
        result = await self.evaluation_agent.process_message(
            message=context,
            enable_tools=True,
        )
 
        return result

    # fmt: on