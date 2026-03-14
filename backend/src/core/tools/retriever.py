"""
Retriever tool for clinical document retrieval.

Wraps RAGService in the BaseTool interface so the agent can autonomously
decide when to retrieve context from the clinical document store.
"""
from typing import Any, Dict, List, Optional

from src.core.tools.base import BaseTool
from src.infrastructure.services.rag import RAGService
from utils.logger import get_logger

logger = get_logger()


class RetrieverTools(BaseTool):
    """
    Agent-facing retrieval tool.

    Exposes two callable tools to the agent:
        - retrieve_clinical_context : General semantic search across all docs
        - retrieve_by_doc_type      : Targeted search filtered to a specific doc type

    Args:
        rag_service: An initialized RAGService instance
        enabled_tools: Optional list of tool names to expose to the agent
    """

    def __init__(
        self,
        rag_service: RAGService,
        enabled_tools: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(enabled_tools=enabled_tools, **kwargs)
        self.rag_service = rag_service

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

    async def _retrieve_clinical_documents_context(
        self,
        query: str,
        top_k: int = 5,
        specialty: str = "",
        **kwargs,
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
                       e.g. "cardiology", "oncology", "pediatrics".
                       Leave empty to search across all specialties.
        """
        try:
            filters = {"specialty": specialty.strip().lower()} if specialty and specialty.strip() else None

            chunks = await self.rag_service.retrieve(query=query, top_k=top_k, filters=filters)

            if not chunks:
                return {
                    "success": True,
                    "results_found": 0,
                    "context": "",
                    "message": "No relevant documents found. Consider rephrasing or broadening the search terms.",
                }

            return {
                "success": True,
                "results_found": len(chunks),
                "context": [chunk.to_dict() for chunk in chunks],
                "message": f"Found {len(chunks)} relevant document chunk(s).",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _retrieve_clinical_documents_by_document_type(
        self,
        query: str,
        doc_type: str,
        top_k: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Retrieve clinical documents filtered to a specific document type.

        Use this tool when you know exactly what category of document you need.
        Prefer this over retrieve_clinical_context when the required document
        type is unambiguous — it produces more targeted, precise results.

        Supported doc_type values:
            - "clinical_guideline"  : Evidence-based treatment guidelines (e.g. NICE, WHO, AHA)
            - "drug_reference"      : Drug monographs, dosage tables, interaction data
            - "patient_record"      : Patient demographics and SOAP reports

        Args:
            query: The clinical question or topic to search for.
            doc_type: The document category to restrict the search to.
            top_k: Number of document chunks to retrieve. Default is 5.
        """
        try:
            if not doc_type or not doc_type.strip():
                return {
                    "success": False,
                    "results_found": 0,
                    "context": "",
                    "message": "doc_type is required for retrieve_by_doc_type.",
                }

            chunks = await self.rag_service.retrieve(
                query=query,
                top_k=top_k,
                filters={"doc_type": doc_type.strip().lower()},
            )

            if not chunks:
                return {
                    "success": True,
                    "results_found": 0,
                    "context": "",
                    "message": f"No '{doc_type}' documents found. Try retrieve_clinical_context for a broader search.",
                }

            return {
                "success": True,
                "results_found": len(chunks),
                "context": [chunk.to_dict() for chunk in chunks],
                "message": f"Found {len(chunks)} '{doc_type}' chunk(s).",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # fmt: on