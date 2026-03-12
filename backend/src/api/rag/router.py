import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from dependency_injector.wiring import Provide, inject

from src.config.dependency_injection.container import Container
from src.infrastructure.services.rag import RAGService
from src.infrastructure.vector_store.chroma import ChromaVectorStore
from src.api.rag.schemas import RAGQueryRequest, RAGQueryResponse, IngestResponse
from utils.ingest import ingest_document
from utils.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/query", response_model=RAGQueryResponse)
@inject
async def query_documents(
    request: RAGQueryRequest,
    rag_service: RAGService = Depends(Provide[Container.rag_service]),
):
    """
    Retrieve clinical documents based on a semantic query.
    """
    try:
        filters = {}
        if request.specialty and request.specialty.strip():
            filters["specialty"] = request.specialty.strip().lower()
        if request.doc_type and request.doc_type.strip():
            filters["doc_type"] = request.doc_type.strip().lower()
            
        # If there are multiple filters, ChromaDB requires an $and operator
        final_filters = None
        if len(filters) == 1:
            final_filters = filters
        elif len(filters) > 1:
            final_filters = {"$and": [{k: v} for k, v in filters.items()]}

        chunks = await rag_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=final_filters,
            score_threshold=request.score_threshold,
        )

        return RAGQueryResponse(
            success=True,
            query=request.query,
            results_found=len(chunks),
            chunks=[chunk.to_dict() for chunk in chunks],
            message=f"Found {len(chunks)} relevant document chunk(s)."
        )

    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=IngestResponse)
@inject
async def upload_document(
    file: UploadFile = File(...),
    doc_type_override: Optional[str] = Form(default=None),
    rag_service: RAGService = Depends(Provide[Container.rag_service]),
    vector_store: ChromaVectorStore = Depends(Provide[Container.vector_store]),
):
    """
    Upload a clinical document (PDF) for ingestion into the ChromaDB vector store.
    """
    # ensure file has a name
    filename = file.filename or "uploaded_document.pdf"
    
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks_ingested = await ingest_document(
            pdf_path=temp_path,
            rag_service=rag_service,
            vector_store=vector_store,
            doc_type_override=doc_type_override,
        )

        success = chunks_ingested > 0
        message = "Successfully ingested document" if success else "Failed to ingest document or no text extracted"

        return IngestResponse(
            success=success,
            filename=filename,
            chunks_ingested=chunks_ingested,
            message=message
        )

    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            temp_path.unlink()
