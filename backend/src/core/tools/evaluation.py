"""
Clinical evaluation tools — hallucination detection, drug checks, completeness.
"""
import json
from typing import Any, Dict, List, Optional
from src.core.tools.base import BaseTool
from src.infrastructure.services.evaluation import EvaluationService
from src.infrastructure.cache.service import CacheService
from utils.logger import get_logger

logger = get_logger()


class EvaluationTools(BaseTool):
    """
    Tools for clinical note evaluation: hallucination detection,
    drug interaction checking, completeness scoring, and guideline alignment.
    """

    def __init__(
        self,
        evaluation_service: EvaluationService,
        cache_service: CacheService,
        enabled_tools: list[str] = None,
        **kwargs
    ) -> None:
        super().__init__(enabled_tools=enabled_tools, **kwargs)
        self.evaluation_service = evaluation_service
        self.cache_service = cache_service

    async def execute(self, tool_name: str, tool_input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
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
            
            # # Ensure result is a dictionary
            # if isinstance(result, tuple):
            #     if len(result) == 3:
            #         success, data, message = result
            #         return {
            #             "success": success,
            #             "data": data,
            #             "message": message
            #         }
            # elif isinstance(result, dict):
            #     return result
            
            # return {"success": True, "data": result}
            result = result if isinstance(result, dict) else {"success": True, "data": result}
 
            # Store each check result in kwargs immediately so aggregate_scores can read them.
            # This also handles the case where the LLM batches all 4 tools in one round:
            # execute() runs them sequentially, so by the time aggregate_scores is called
            # the three check results are already stored.
            result_key_map = {
                "check_hallucinations": "_hallucination_result",
                "check_drug_interactions": "_drug_result",
                "check_guideline_alignment": "_guideline_result",
            }
            if tool_name in result_key_map:
                self.kwargs[result_key_map[tool_name]] = result
                logger.debug(f"Stored {result_key_map[tool_name]} in EvaluationTools.kwargs")
 
            # If aggregate_scores fired before the check results were stored (can happen
            # when the LLM batches all 4 tools in the same response and execute() processes
            # them in list order with aggregate_scores last), re-run it now that all results
            # are available. The warning log is the signal.
            if tool_name == "aggregate_scores" and result.get("success"):
                scores = result.get("scores", {})
                # If all three inputs were empty dicts, the results weren't ready — re-run
                h = self.kwargs.get("_hallucination_result")
                d = self.kwargs.get("_drug_result")
                g = self.kwargs.get("_guideline_result")
                if not scores.get("completeness") and (h or d or g):
                    logger.debug("aggregate_scores re-running with now-available check results")
                    result = await method(**tool_input_with_context)
                    result = result if isinstance(result, dict) else {"success": True, "data": result}
 
            return result
            
        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to execute {tool_name}"
            }

    # fmt: off

    async def _check_hallucinations(self, **kwargs) -> Dict[str, Any]:
        """
        Detect claims in the SOAP note not grounded in the consultation transcript.
        Identifies fabricated symptoms, vitals, medications, or diagnoses.
        Returns hallucination flags, overall risk level, and completeness score.
        Call this first before other evaluation checks.
        """

        transcript = kwargs.get("transcript") or self.kwargs.get("transcript", "")
        session_id = kwargs.get("session_id")
        soap_note = await self.cache_service.get(f"soap:{session_id}") or {}
        conditions = soap_note.get("conditions_detected", [])
 
        return await self.evaluation_service.check_hallucinations(
            soap_note=soap_note,
            transcript=transcript,
            session_id=session_id,
        )
 
    async def _check_drug_interactions(self, **kwargs) -> Dict[str, Any]:
        """
        Cross-reference all medications mentioned in the SOAP note against a known
        drug interaction database. Returns interaction alerts, severity levels,
        and a drug safety score. Call this after check_hallucinations.
        """
        session_id = kwargs.get("session_id")
        soap_note = await self.cache_service.get(f"soap:{session_id}") or {}
        medications = soap_note.get("medications_mentioned", [])
 
        return await self.evaluation_service.check_drug_interactions(
            medications=medications,
        )
 
    async def _check_guideline_alignment(self, **kwargs) -> Dict[str, Any]:
        """
        Check whether the documented assessment and plan aligns with established
        clinical guidelines for the patient's conditions. Returns an alignment
        score, identified gaps, and improvement suggestions.
        Call this after check_drug_interactions.
        """
        session_id = kwargs.get("session_id")
        soap_note = await self.cache_service.get(f"soap:{session_id}") or {}
        conditions = soap_note.get("conditions_detected", [])
 
        return await self.evaluation_service.check_guideline_alignment(
            soap_note=soap_note,
            conditions=conditions,
        )
 
    async def _aggregate_scores(
        self,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Aggregate results from all evaluation checks into a unified dashboard payload.
        Pass the exact outputs from check_hallucinations, check_drug_interactions,
        and check_guideline_alignment as arguments.
        Call this last after all three checks are complete.
 
        Args:
            hallucination_result: Output from check_hallucinations
            drug_result: Output from check_drug_interactions
            guideline_result: Output from check_guideline_alignment
        """
        session_id = kwargs.get("session_id")

        # Read results stored by execute() after each check ran — never trust
        # whatever the LLM passes as arguments here, as it tends to summarize.
        hallucination_result = kwargs.get("_hallucination_result") or {}
        drug_result = kwargs.get("_drug_result") or {}
        guideline_result = kwargs.get("_guideline_result") or {}

        logger.debug(f"Hallucination result: {hallucination_result}, Drug result: {drug_result}, Guideline result: {guideline_result}")
 
        if not any([hallucination_result, drug_result, guideline_result]):
            logger.warning("aggregate_scores called but no check results stored in kwargs yet")
 
        return await self.evaluation_service.aggregate_scores(
            hallucination_result=hallucination_result or {},
            drug_result=drug_result or {},
            guideline_result=guideline_result or {},
            session_id=session_id,
        )

    # fmt: on