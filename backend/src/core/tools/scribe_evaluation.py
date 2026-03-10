"""
Evaluation tools for ScribeAgent.

Single tool: _evaluate_consultation — calls EvaluationService which
internally runs its own EvaluationAgent with all evaluation checks.

ScribeAgent calls this after SOAP generation is complete.
All evaluation logic lives in EvaluationService — zero impl here.
"""
from typing import Any, Dict, List, Optional

from src.core.tools.base import BaseTool
from src.infrastructure.services.evaluation import EvaluationService
from utils.logger import get_logger

logger = get_logger()


class EvaluationTools(BaseTool):
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

    def __init__(self, evaluation_service: EvaluationService, enabled_tools: list[str] = None, **kwargs):
        super().__init__(enabled_tools=enabled_tools, **kwargs)
        self.evaluation_service = evaluation_service

    async def _evaluate_consultation(self) -> Dict[str, Any]:
        """
        Run a full clinical evaluation of the generated SOAP note.
        Checks for hallucinations (claims not in transcript), drug interactions,
        clinical guideline alignment, and documentation completeness.
        Returns aggregated scores ready for the physician dashboard.
        Call this after the SOAP note has been generated.
        """
        session_id = kwargs.get("session_id")
        transcript = kwargs.get("transcript", "")
        soap_note = kwargs.get("soap_note", {})
        conditions = kwargs.get("conditions", [])

        # this will eventually be its own agent
        return await self.evaluation_service.run_full_evaluation(
            soap_note=soap_note,
            transcript=transcript,
            conditions=conditions,
            session_id=session_id,
        )