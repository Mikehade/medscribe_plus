"""
Retriever tool for clinical document retrieval.

Wraps RAGService in the BaseTool interface so the agent can autonomously
decide when to retrieve context from the clinical document store.

Follows the exact pattern established in src/core/tools/base.py —
tool methods are prefixed with a single underscore, and execute()
dispatches by tool name.
"""
from typing import Any, Dict, List, Optional

from src.core.tools.base import BaseTool
from src.infrastructure.services.rag import RAGService
from utils.logger import get_logger

logger = get_logger()


class RetrieverTool(BaseTool):
    """
    Agent-facing retrieval tool.

    Exposes two callable tools to the agent:
        - retrieve_clinical_context  : General semantic search across all docs
        - retrieve_by_doc_type       : Targeted search filtered to a specific doc type

    Both tools are discovered automatically by BaseTool.generate_bedrock_config()
    because their method names carry a single underscore prefix.

    Usage (wiring into the agent):
        rag_service = RAGService(vector_store=chroma_store)
        retriever = RetrieverTool(
            rag_service=rag_service,
            enabled_tools=["retrieve_clinical_context", "retrieve_by_doc_type"]
        )
        registry = ToolRegistry(tool_classes=[retriever, ...other tools...])
    """

    def __init__(
        self,
        rag_service: RAGService,
        enabled_tools: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize retriever tool.

        Args:
            rag_service: An initialized RAGService instance
            enabled_tools: Optional list of tool names to expose to the agent.
                           Defaults to all tools on this class.
            **kwargs: Passed through to BaseTool
        """
        super().__init__(enabled_tools=enabled_tools, **kwargs)
        self.rag_service = rag_service

    # ------------------------------------------------------------------
    # Tool methods — discovered automatically by BaseTool via underscore prefix
    # ------------------------------------------------------------------

    async def _retrieve_clinical_context(
        self,
        query: str,
        top_k: int = 5,
        specialty: str = "",
    ) -> Dict[str, Any]:
        """
        Retrieve relevant clinical documents based on a semantic query.

        Use this tool when you need evidence-based clinical information such as
        treatment protocols, drug dosages, diagnostic criteria, clinical guidelines,
        or any medical reference material to support your response.

        Args:
            query: The clinical question or topic to search for.
                   Be specific — e.g. "first-line treatment for community acquired pneumonia"
                   rather than just "pneumonia treatment".
            top_k: Number of document chunks to retrieve. Default is 5.
                   Increase to 8-10 for complex multi-faceted queries.
            specialty: Optional medical specialty to filter results.
                       e.g. "cardiology", "oncology", "pediatrics", "icu".
                       Leave empty to search across all specialties.

        Returns:
            Dict containing retrieved context chunks and metadata.
        """
        try:
            # Build metadata filter if specialty is specified
            filters = None
            if specialty and specialty.strip():
                filters = {"specialty": specialty.strip().lower()}

            chunks = await self.rag_service.retrieve(
                query=query,
                top_k=top_k,
                filters=filters,
            )

            if not chunks:
                return {
                    "success": True,
                    "query": query,
                    "results_found": 0,
                    "context": "",
                    "chunks": [],
                    "message": (
                        "No relevant documents found for this query. "
                        "Consider rephrasing or broadening the search terms."
                    ),
                }

            return {
                "success": True,
                "query": query,
                "results_found": len(chunks),
                "context": self._format_context_for_agent(chunks),
                "chunks": [chunk.to_dict() for chunk in chunks],
                "message": f"Found {len(chunks)} relevant document chunk(s).",
            }

        except Exception as e:
            logger.error(f"retrieve_clinical_context failed: {e}", exc_info=True)
            return {
                "success": False,
                "query": query,
                "results_found": 0,
                "context": "",
                "chunks": [],
                "message": f"Retrieval failed: {str(e)}",
            }

    async def _retrieve_by_doc_type(
        self,
        query: str,
        doc_type: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrieve clinical documents filtered to a specific document type.

        Use this tool when you know exactly what category of document you need.
        Prefer this over retrieve_clinical_context when the required document
        type is unambiguous — it produces more targeted, precise results.

        Supported doc_type values:
            - "clinical_guideline"   : Evidence-based treatment guidelines (e.g. NICE, WHO, AHA)
            - "drug_reference"       : Drug monographs, dosage tables, interaction data
            - "protocol"             : Hospital or departmental care protocols
            - "diagnostic_criteria"  : DSM, ICD, or other diagnostic classification criteria
            - "patient_education"    : Patient-facing educational materials
            - "research_summary"     : Summarised clinical trial or systematic review findings

        Args:
            query: The clinical question or topic to search for.
            doc_type: The document category to restrict the search to.
                      Must be one of the supported values listed above.
            top_k: Number of document chunks to retrieve. Default is 5.

        Returns:
            Dict containing retrieved context chunks and metadata.
        """
        try:
            if not doc_type or not doc_type.strip():
                return {
                    "success": False,
                    "query": query,
                    "doc_type": doc_type,
                    "results_found": 0,
                    "context": "",
                    "chunks": [],
                    "message": "doc_type is required for retrieve_by_doc_type.",
                }

            filters = {"doc_type": doc_type.strip().lower()}

            chunks = await self.rag_service.retrieve(
                query=query,
                top_k=top_k,
                filters=filters,
            )

            if not chunks:
                return {
                    "success": True,
                    "query": query,
                    "doc_type": doc_type,
                    "results_found": 0,
                    "context": "",
                    "chunks": [],
                    "message": (
                        f"No '{doc_type}' documents found for this query. "
                        f"Try retrieve_clinical_context for a broader search."
                    ),
                }

            return {
                "success": True,
                "query": query,
                "doc_type": doc_type,
                "results_found": len(chunks),
                "context": self._format_context_for_agent(chunks),
                "chunks": [chunk.to_dict() for chunk in chunks],
                "message": f"Found {len(chunks)} '{doc_type}' chunk(s).",
            }

        except Exception as e:
            logger.error(f"retrieve_by_doc_type failed: {e}", exc_info=True)
            return {
                "success": False,
                "query": query,
                "doc_type": doc_type,
                "results_found": 0,
                "context": "",
                "chunks": [],
                "message": f"Retrieval failed: {str(e)}",
            }

    # ------------------------------------------------------------------
    # BaseTool interface — required abstract method
    # ------------------------------------------------------------------

    async def execute(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Dispatch tool execution by name.
        Called by ToolRegistry.execute_tool() during the agent's tool loop.

        Args:
            tool_name: Name of the tool to run (without underscore prefix)
            tool_input: Parameters dict from the model's tool call

        Returns:
            Tool result dict — always includes "success" and "message" keys
        """
        method = self.get_tool_method(tool_name)

        if not method:
            logger.error(f"RetrieverTool: unknown tool '{tool_name}'")
            return {
                "success": False,
                "message": f"Unknown retriever tool: '{tool_name}'",
            }

        logger.info(f"RetrieverTool executing: '{tool_name}' | input: {tool_input}")

        return await method(**tool_input)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_context_for_agent(self, chunks) -> str:
        """
        Format retrieved chunks into a clean, readable context block
        for injection into the agent's reasoning context.

        Each chunk is clearly delimited with its source and type so the
        agent can attribute its response accurately.

        Args:
            chunks: List of RetrievedChunk objects from RAGService

        Returns:
            Formatted string with all chunks concatenated
        """
        formatted_chunks = []

        for i, chunk in enumerate(chunks, start=1):
            formatted_chunks.append(
                f"[DOCUMENT {i}]\n"
                f"Source   : {chunk.source}\n"
                f"Type     : {chunk.doc_type}\n"
                f"Relevance: {chunk.score}\n"
                f"Content  :\n{chunk.content}\n"
            )

        return "\n" + ("-" * 60 + "\n").join(formatted_chunks)