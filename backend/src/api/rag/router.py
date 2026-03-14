"""
RAG API router.

Endpoints:
    POST /rag/query   — semantic retrieval over ingested documents
    POST /rag/upload  — ingest a PDF into the vector store

The router depends on RAGService only. All PDF parsing, chunking,
embedding, and storage are delegated to the service layer — the router
handles HTTP concerns (request validation, file I/O, error responses).
"""
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from dependency_injector.wiring import Provide, inject

from src.config.dependency_injection.container import Container
from src.infrastructure.services.rag import RAGService
from src.api.rag.schemas import IngestResponse, RAGQueryRequest, RAGQueryResponse
from utils.ingest import detect_doc_type, extract_metadata, extract_text_from_pdf
from utils.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/rag", tags=["RAG"])

# Temp directory for PDF uploads — cleaned up after ingestion
_TEMP_DIR = Path("./temp_uploads")


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

@router.post("/query", response_model=RAGQueryResponse)
@inject
async def query_documents(
    request: RAGQueryRequest,
    rag_service: RAGService = Depends(Provide[Container.rag_service]),
):
    """
    Retrieve clinical document chunks using semantic similarity.

    Optionally filter by `specialty` and/or `doc_type`. Both fields are
    optional — omit them to search across the entire collection.
    """
    try:
        # Build metadata filters
        filters: dict = {}
        if request.specialty and request.specialty.strip():
            filters["specialty"] = request.specialty.strip().lower()
        if request.doc_type and request.doc_type.strip():
            filters["doc_type"] = request.doc_type.strip().lower()

        final_filters: Optional[dict] = None
        if len(filters) == 1:
            final_filters = filters
        elif len(filters) > 1:
            # ChromaDB requires $and for multiple filters
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
            message=f"Found {len(chunks)} relevant document chunk(s).",
        )

    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Upload / ingest endpoint
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=IngestResponse)
@inject
async def upload_document(
    file: UploadFile = File(...),
    doc_type_override: Optional[str] = Form(default=None),
    rag_service: RAGService = Depends(Provide[Container.rag_service]),
):
    """
    Upload a clinical PDF and ingest it into the vector store.

    The router is responsible for:
        - Saving the upload to a temp file
        - Extracting text from the PDF (utils helper)
        - Detecting doc type and extracting metadata (utils helpers)
        - Delegating embedding + upsert to RAGService

    The router is NOT responsible for:
        - Chunking (handled inside RAGService.ingest_document)
        - Embedding (handled by the injected embedding model)
        - Vector store interaction (handled by the injected vector store)
    """
    filename = file.filename or "uploaded_document.pdf"
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = _TEMP_DIR / filename

    try:
        # Save upload to disk temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- PDF parsing & metadata extraction (utils layer) ---
        text = extract_text_from_pdf(temp_path)
        if not text.strip():
            return IngestResponse(
                success=False,
                filename=filename,
                chunks_ingested=0,
                message="No text could be extracted from the uploaded PDF.",
            )

        doc_type = doc_type_override or detect_doc_type(text)
        metadata = extract_metadata(
            text=text,
            doc_type=doc_type,
            source_filename=filename,
        )

        # Stable prefix for idempotent upserts (re-ingesting same file
        # overwrites old chunks rather than creating duplicates)
        doc_id_prefix = (
            f"{temp_path.stem}_{doc_type}".replace(" ", "_").lower()
        )

        # --- Embedding + upsert (service layer) ---
        chunks_ingested = await rag_service.ingest_document(
            text=text,
            metadata=metadata,
            doc_id_prefix=doc_id_prefix,
        )

        success = chunks_ingested > 0
        message = (
            f"Successfully ingested {chunks_ingested} chunks."
            if success
            else "Ingestion produced no chunks — check PDF content."
        )

        return IngestResponse(
            success=success,
            filename=filename,
            chunks_ingested=chunks_ingested,
            message=message,
        )

    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path.exists():
            temp_path.unlink()