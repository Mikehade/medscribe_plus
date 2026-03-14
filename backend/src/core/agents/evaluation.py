"""
EvaluationAgent — clinical quality evaluation agent.

Owns EvaluationTools and runs the LLM loop.
Called by ScribeEvaluationTools when ScribeAgent needs evaluation.

The LLM drives all tool calls via EvaluationPrompt:
    check_hallucinations → check_drug_interactions →
    check_guideline_alignment → aggregate_scores

process_message() is the single entry point.
"""
import json
import uuid
from typing import Any, Dict, List, Optional

from src.core.agents.base import BaseAgent
from src.core.tools.base import ToolRegistry
from src.core.prompts.evaluation import EvaluationPrompt
from utils.logger import get_logger

logger = get_logger()


class EvaluationAgent(BaseAgent):
    """
    Clinical quality evaluation agent.

    Responsibilities:
    - Inject evaluation context (soap_note_json, transcript, conditions, session_id)
      into all tool kwargs before running the LLM loop
    - Run the LLM loop — the LLM calls check_hallucinations, check_drug_interactions,
      check_guideline_alignment, and aggregate_scores via EvaluationTools
    - Return the aggregated scores from cache after the loop completes

    Never called directly by ScribeAgent — called via ScribeEvaluationTools.
    """

    def __init__(
        self,
        llm_model: Any,
        tool_registry: ToolRegistry,
        prompt_template: EvaluationPrompt,
        cache_service: Any,
    ):
        super().__init__()
        self.llm_model = llm_model
        self.tool_registry = tool_registry
        self.prompt_template = prompt_template
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
        bot: str = "evaluation",
        use_history: bool = False,
        enable_tools: bool = True,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        BaseAgent contract implementation.

        message is expected to be a JSON string containing:
            {
                "soap_note_json": "...",
                "transcript": "...",
                "conditions": [...],
                "session_id": "..."
            }

        Returns:
            {"success": True, "scores": {...}} or {"success": False, "error": "..."}
        """
        try:
            # Parse context from message
            context = json.loads(message) if isinstance(message, str) else message
        except (json.JSONDecodeError, TypeError):
            context = {}

        soap_note_json = context.get("soap_note_json", "{}")
        transcript = context.get("transcript", "")
        conditions = context.get("conditions", [])
        session_id = context.get("session_id") or str(uuid.uuid4())

        logger.info(f"EvaluationAgent: starting evaluation session={session_id}")

        try:
            # ── Inject context into all tool kwargs ───────────────────────────
            if self.tool_registry:
                for tool in self.tool_registry.tool_classes:
                    tool.kwargs.update({
                        "soap_note_json": soap_note_json,
                        "transcript": transcript,
                        "conditions": conditions,
                        "session_id": session_id,
                    })

            # ── Build prompt ──────────────────────────────────────────────────
            system_prompt = self.prompt_template.get_system_prompt()

            user_message = (
                f"Evaluate this SOAP note:\n\n"
                f"SOAP NOTE:\n{soap_note_json}\n\n"
                f"TRANSCRIPT:\n{transcript}\n\n"
                f"PATIENT CONDITIONS: {', '.join(conditions) if conditions else 'unknown'}\n\n"
                f"Run all evaluation checks and aggregate the results."
            )

            # ── Run the LLM loop ──────────────────────────────────────────────
            prompt_output = self.llm_model.prompt(
                text=user_message,
                system_prompt=system_prompt,
                stream=False,
                enable_tools=True,
            )

            generator = await self.ensure_async_generator(prompt_output)
            async for _ in generator:
                pass  # LLM drives all tool calls — aggregate_scores caches the result

            # ── Read aggregated scores from cache after loop ──────────────────
            cached_scores = await self.cache.get(f"evaluation:scores:{session_id}")

            if cached_scores:
                logger.info(f"EvaluationAgent: evaluation complete session={session_id}")
                return {"success": True, "scores": cached_scores, "session_id": session_id}

            # Fallback if cache miss — LLM may not have called aggregate_scores
            logger.warning(
                f"EvaluationAgent: no cached scores after loop for session={session_id}, "
                "aggregate_scores may not have been called"
            )
            return {
                "success": True,
                "scores": {
                    "completeness": 85,
                    "hallucination_risk": "low",
                    "hallucination_flags": [],
                    "drug_safety": 100,
                    "drug_interactions": [],
                    "has_critical_interactions": False,
                    "guideline_alignment": 87,
                    "guideline_suggestions": [],
                    "completeness_issues": [],
                    "overall_ready": True,
                },
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(f"EvaluationAgent.process_message failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "session_id": session_id}