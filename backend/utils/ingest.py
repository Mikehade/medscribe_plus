"""
Clinical document ingestion utilities.

This module owns the document-processing pipeline:
    - PDF text extraction
    - Document type detection
    - Metadata extraction
    - Sentence-aware text chunking
    - Chunk ID + metadata preparation

It does NOT own embedding or vector store operations. Those concerns
belong to RAGService. The CLI entry point (ingest_directory) composes
everything by calling RAGService.ingest_document after preparing text.

Three supported document types:
    patient_record      : Patient demographics + SOAP report
    clinical_guideline  : Condition-based treatment guidelines
    drug_reference      : Drug monographs with interactions/contraindications

Run from the project root:
    python -m utils.ingest --docs_dir ./documents --collection clinical_docs
"""
import argparse
import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 5
CHUNK_OVERLAP = 2
MIN_CHUNK_CHARS = 80

DOC_TYPE_SIGNATURES = {
    "patient_record": [
        "patient id", "date of birth", "chronic condition",
        "subjective", "objective", "assessment", "plan", "soap",
        "general outpatient", "chief complaint",
    ],
    "clinical_guideline": [
        "diagnostic criteria", "first-line treatment", "second-line treatment",
        "referral criteria", "red flag", "recommended investigations",
        "lifestyle recommendations", "risk factors", "definition",
    ],
    "drug_reference": [
        "drug name", "drug interactions", "contraindications",
        "side effects", "indications", "class:", "dosage",
        "mechanism of action",
    ],
}

PATIENT_PATTERNS = {
    "patient_id":          r"patient\s*id[:\s]+([A-Z0-9\-]+)",
    "date":                r"date[:\s]+(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\w+ \d+, \d{4})",
    "location":            r"location[:\s]+(.+?)(?:\n|age|$)",
    "age":                 r"age[:\s]+(\d+)",
    "chronic_conditions":  r"chronic condition[s]?[:\s]+(.+?)(?:\n\n|\Z)",
}

GUIDELINE_PATTERNS = {
    "condition":  r"condition[:\s]+(.+?)(?:\n|definition|$)",
    "specialty":  r"specialty[:\s]+(.+?)(?:\n|$)",
}

DRUG_PATTERNS = {
    "drug_name":   r"drug\s*name[:\s]+(.+?)(?:\n|class|$)",
    "drug_class":  r"class[:\s]+(.+?)(?:\n|indication|$)",
}


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file using pdfplumber.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text string, or empty string on failure
    """
    try:
        full_text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text.strip())

        combined = "\n\n".join(full_text)
        combined = re.sub(r"\n{3,}", "\n\n", combined)

        logger.info(f"Extracted {len(combined)} chars from: {pdf_path.name}")
        return combined

    except Exception as e:
        logger.error(f"PDF extraction failed for {pdf_path}: {e}", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Document Type Detection
# ---------------------------------------------------------------------------

def detect_doc_type(text: str) -> str:
    """
    Detect document type by scoring keyword matches against the text.

    Args:
        text: Extracted PDF text

    Returns:
        One of: "patient_record", "clinical_guideline", "drug_reference", "general"
    """
    lowered = text.lower()
    scores: Dict[str, int] = {
        doc_type: sum(1 for kw in keywords if kw in lowered)
        for doc_type, keywords in DOC_TYPE_SIGNATURES.items()
    }

    best_type = max(scores, key=lambda k: scores[k])

    if scores[best_type] == 0:
        logger.warning("Could not detect document type — defaulting to 'general'")
        return "general"

    logger.info(f"Detected doc_type='{best_type}' | scores: {scores}")
    return best_type


# ---------------------------------------------------------------------------
# Metadata Extraction
# ---------------------------------------------------------------------------

def extract_metadata(text: str, doc_type: str, source_filename: str) -> Dict[str, Any]:
    """
    Extract structured metadata from document text.

    Args:
        text: Full document text
        doc_type: Detected or provided document type
        source_filename: Original PDF filename

    Returns:
        Flat metadata dict safe for ChromaDB storage
    """
    metadata: Dict[str, Any] = {
        "source": source_filename,
        "doc_type": doc_type,
    }

    lowered = text.lower()

    if doc_type == "patient_record":
        for field, pattern in PATIENT_PATTERNS.items():
            match = re.search(pattern, lowered, re.IGNORECASE | re.DOTALL)
            if match:
                metadata[field] = match.group(1).strip().replace("\n", " ")

        if "cardio" in lowered:
            metadata["specialty"] = "cardiology"
        elif "diabetes" in lowered or "endocrin" in lowered:
            metadata["specialty"] = "endocrinology"
        elif "hypertension" in lowered:
            metadata["specialty"] = "internal_medicine"

    elif doc_type == "clinical_guideline":
        for field, pattern in GUIDELINE_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata[field] = match.group(1).strip()

        if "specialty" not in metadata:
            if any(kw in lowered for kw in ["hypertension", "heart", "cardiac"]):
                metadata["specialty"] = "cardiology"
            elif any(kw in lowered for kw in ["diabetes", "glucose", "insulin"]):
                metadata["specialty"] = "endocrinology"
            elif any(kw in lowered for kw in ["pneumonia", "respiratory", "asthma"]):
                metadata["specialty"] = "pulmonology"

    elif doc_type == "drug_reference":
        for field, pattern in DRUG_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata[field] = match.group(1).strip()

    logger.debug(f"Extracted metadata: {metadata}")
    return metadata


# ---------------------------------------------------------------------------
# Sentence-Aware Sliding Window Chunker
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex — no external NLP downloads.

    Args:
        text: Raw document text

    Returns:
        List of sentence strings
    """
    sentences = []
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if re.match(r'^[A-Z][A-Z\s]+[:\-]', line) or line.isupper():
            if len(line) >= MIN_CHUNK_CHARS // 2:
                sentences.append(line)
            continue

        for part in sentence_pattern.split(line):
            part = part.strip()
            if len(part) >= MIN_CHUNK_CHARS // 2:
                sentences.append(part)

    return sentences


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping sentence-aware chunks.

    Args:
        text: Full document text
        chunk_size: Number of sentences per chunk
        overlap: Sentences shared between consecutive chunks

    Returns:
        List of chunk strings, each >= MIN_CHUNK_CHARS
    """
    sentences = _split_sentences(text)

    if not sentences:
        logger.warning("No sentences extracted from text — check PDF extraction")
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0

    while start < len(sentences):
        end = min(start + chunk_size, len(sentences))
        chunk = " ".join(sentences[start:end]).strip()

        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)

        if end == len(sentences):
            break

        start += step

    logger.debug(
        f"Chunked into {len(chunks)} windows (size={chunk_size}, overlap={overlap})"
    )
    return chunks


# ---------------------------------------------------------------------------
# Document Preparation
# ---------------------------------------------------------------------------

def prepare_documents(
    chunks: List[str],
    metadata: Dict[str, Any],
    doc_id_prefix: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Assign stable IDs and enrich metadata per chunk.

    Chunk IDs follow the format:
        {doc_id_prefix}_chunk_{index}
    e.g. "ng-001_patient_record_chunk_0"

    Re-ingesting the same document produces the same IDs, so ChromaDB
    upsert overwrites stale chunks cleanly.

    Args:
        chunks: List of text chunks
        metadata: Base metadata dict for the document
        doc_id_prefix: Stable prefix derived from filename + doc_type

    Returns:
        (chunk_ids, enriched_metadatas)
    """
    total = len(chunks)
    ids = []
    metadatas = []

    for i in range(total):
        ids.append(f"{doc_id_prefix}_chunk_{i}")
        metadatas.append({**metadata, "chunk_index": i, "total_chunks": total})

    return ids, metadatas


# ---------------------------------------------------------------------------
# CLI entry point — composes utils + RAGService for batch ingestion
# ---------------------------------------------------------------------------

async def ingest_directory(
    docs_dir: str,
    collection_name: str,
    persist_directory: str,
    aws_access_key: str,
    aws_secret_key: str,
    aws_region: str = "us-east-1",
) -> None:
    """
    Ingest all PDFs in a directory using the full service stack.

    Initialises infrastructure directly (no DI container) since this
    runs as a CLI script outside the FastAPI application context.

    Args:
        docs_dir: Path to directory containing PDF files
        collection_name: ChromaDB collection name
        persist_directory: Local ChromaDB storage path
        aws_access_key: AWS access key for Bedrock
        aws_secret_key: AWS secret key for Bedrock
        aws_region: AWS region for Bedrock Nova Embed
    """
    # Lazy imports — only needed for CLI usage
    from src.infrastructure.embedding_models.bedrock import BedrockEmbeddingModel
    from src.infrastructure.services.rag import RAGService
    from src.infrastructure.vector_store.chroma import ChromaVectorStore

    docs_path = Path(docs_dir)

    if not docs_path.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        return

    pdf_files = sorted(docs_path.glob("**/*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files found in: {docs_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) to ingest")

    # Initialise infrastructure
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    await vector_store.initialize()

    embedding_model = BedrockEmbeddingModel(
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        region_name=aws_region,
    )

    rag_service = RAGService(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    # Process each PDF
    results: Dict[str, int] = {}

    for pdf_path in pdf_files:
        logger.info(f"\n{'='*60}\nIngesting: {pdf_path.name}\n{'='*60}")

        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            logger.error(f"No text extracted from {pdf_path.name} — skipping")
            results[pdf_path.name] = 0
            continue

        doc_type = detect_doc_type(text)
        metadata = extract_metadata(
            text=text,
            doc_type=doc_type,
            source_filename=pdf_path.name,
        )
        doc_id_prefix = f"{pdf_path.stem}_{doc_type}".replace(" ", "_").lower()

        count = await rag_service.ingest_document(
            text=text,
            metadata=metadata,
            doc_id_prefix=doc_id_prefix,
        )
        results[pdf_path.name] = count

    # Summary
    total_chunks = sum(results.values())
    collection_count = await vector_store.count()

    logger.info(f"\n{'='*60}\nINGESTION COMPLETE\n{'='*60}")
    for filename, count in results.items():
        logger.info(f"  {'✓' if count > 0 else '✗'}  {filename}: {count} chunks")

    logger.info(f"\nTotal chunks ingested : {total_chunks}")
    logger.info(f"Total in collection   : {collection_count}")
    logger.info(f"Collection            : '{collection_name}'")
    logger.info(f"Storage               : {persist_directory}")

    await vector_store.close()


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(
        description="Ingest clinical PDFs into ChromaDB via Nova Embed"
    )
    parser.add_argument("--docs_dir", default="./documents")
    parser.add_argument("--collection", default="clinical_docs")
    parser.add_argument("--persist_dir", default="./chroma_db")
    parser.add_argument("--aws_region", default="us-east-1")
    args = parser.parse_args()

    asyncio.run(
        ingest_directory(
            docs_dir=args.docs_dir,
            collection_name=args.collection,
            persist_directory=args.persist_dir,
            aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_region=args.aws_region,
        )
    )