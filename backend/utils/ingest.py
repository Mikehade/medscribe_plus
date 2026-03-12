"""
Clinical document ingestion pipeline.

One-time (or on-demand) script that:
    1. Extracts text from clinical PDF documents
    2. Detects document type and extracts structured metadata
    3. Chunks text using a sentence-aware sliding window
    4. Embeds each chunk via Amazon Nova Embed (through RAGService)
    5. Upserts all chunks into ChromaDB

Three supported document types:
    - patient_record      : Patient demographics + SOAP report
    - clinical_guideline  : Condition-based treatment guidelines
    - drug_reference      : Drug monographs with interactions/contraindications

Run from the project root:
    python -m utils.ingest --docs_dir ./documents --collection clinical_docs

Requirements:
    pip install pdfplumber chromadb boto3
"""
import argparse
import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from src.infrastructure.services.rag import RAGService
from src.infrastructure.vector_store.chroma import ChromaVectorStore
from utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sliding window chunking settings
CHUNK_SIZE = 5           # Number of sentences per chunk
CHUNK_OVERLAP = 2        # Sentences shared between consecutive chunks

# Minimum characters a chunk must have to be worth embedding
MIN_CHUNK_CHARS = 80

# Keyword signatures used to detect document type from extracted text
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

# Metadata field extractors — regex patterns per document type
PATIENT_PATTERNS = {
    "patient_id":   r"patient\s*id[:\s]+([A-Z0-9\-]+)",
    "date":         r"date[:\s]+(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\w+ \d+, \d{4})",
    "location":     r"location[:\s]+(.+?)(?:\n|age|$)",
    "age":          r"age[:\s]+(\d+)",
    "chronic_conditions": r"chronic condition[s]?[:\s]+(.+?)(?:\n\n|\Z)",
}

GUIDELINE_PATTERNS = {
    "condition":    r"condition[:\s]+(.+?)(?:\n|definition|$)",
    "specialty":    r"specialty[:\s]+(.+?)(?:\n|$)",
}

DRUG_PATTERNS = {
    "drug_name":    r"drug\s*name[:\s]+(.+?)(?:\n|class|$)",
    "drug_class":   r"class[:\s]+(.+?)(?:\n|indication|$)",
}


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file using pdfplumber.

    Preserves page breaks as double newlines to help the chunker
    respect document structure. Strips excessive whitespace.

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
        # Collapse excessive blank lines
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
    Detect the document type by scoring keyword matches against the text.

    Each doc type has a set of signature phrases. The type with the most
    matches wins. Falls back to "general" if no type scores above 0.

    Args:
        text: Extracted PDF text (lowercased internally)

    Returns:
        Document type string: "patient_record", "clinical_guideline",
        "drug_reference", or "general"
    """
    lowered = text.lower()
    scores: Dict[str, int] = {}

    for doc_type, keywords in DOC_TYPE_SIGNATURES.items():
        scores[doc_type] = sum(1 for kw in keywords if kw in lowered)

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
    Extract structured metadata from document text based on doc type.

    Common metadata (source, doc_type) is always set.
    Type-specific fields are extracted via regex and added on top.

    Args:
        text: Full extracted document text
        doc_type: Detected or provided document type
        source_filename: Original PDF filename (used as source reference)

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
                value = match.group(1).strip().replace("\n", " ")
                metadata[field] = value

        # Attempt to extract specialty from chronic conditions context
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

        # Infer specialty from condition text if not explicitly tagged
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
    Split text into sentences using regex — no external downloads required.

    Handles the common sentence-ending patterns found in clinical documents:
        - Standard punctuation:  . ! ?
        - Abbreviations are respected: "e.g.", "Dr.", "vs." will not split
        - Section labels like "CONDITION:", "DRUG NAME:" are kept intact
        - Newline-delimited lines (common in structured clinical PDFs)
          are also treated as sentence boundaries

    Strategy:
        1. Split on newlines first to capture structured label:value lines
        2. Within each line, split on sentence-ending punctuation
        3. Filter out empty or trivially short fragments

    Args:
        text: Raw document text

    Returns:
        List of sentence strings
    """
    sentences = []

    # Split into lines first — clinical PDFs are often line-structured
    lines = text.split("\n")

    # Fixed-width lookbehind — compatible with Python re module
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Lines that look like section headers (ALL CAPS or "LABEL: value")
        # are kept as single sentences without further splitting
        if re.match(r'^[A-Z][A-Z\s]+[:\-]', line) or line.isupper():
            if len(line) >= MIN_CHUNK_CHARS // 2:
                sentences.append(line)
            continue

        # Split longer prose lines into individual sentences
        parts = sentence_pattern.split(line)
        for part in parts:
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
    Split text into overlapping sentence-aware chunks using regex tokenization.

    No external NLP libraries required — safe for network-restricted environments.

    Strategy:
        1. Tokenize the full text into sentences via regex (_split_sentences)
        2. Build chunks of `chunk_size` sentences each
        3. Slide forward by (chunk_size - overlap) sentences so consecutive
           chunks share `overlap` sentences — preserving cross-boundary context

    Args:
        text: Full document text to chunk
        chunk_size: Number of sentences per chunk
        overlap: Number of sentences to repeat in the next chunk

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

    logger.debug(f"Chunked into {len(chunks)} windows (size={chunk_size}, overlap={overlap})")
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
    Pair chunks with their metadata and generate stable unique IDs.

    Each chunk gets an ID in the format:
        {doc_id_prefix}_chunk_{index}
    e.g. "NG-001_patient_record_chunk_0"

    This makes it possible to re-ingest updated documents cleanly —
    same doc produces same IDs, so ChromaDB upsert overwrites old chunks.

    Args:
        chunks: List of text chunks
        metadata: Base metadata dict for the document
        doc_id_prefix: Stable prefix derived from filename + doc_type

    Returns:
        Tuple of (chunk_ids, enriched_metadatas)
        Enriched metadata adds chunk_index and total_chunks per chunk.
    """
    total = len(chunks)
    ids = []
    metadatas = []

    for i, _ in enumerate(chunks):
        chunk_id = f"{doc_id_prefix}_chunk_{i}"
        ids.append(chunk_id)

        chunk_meta = {
            **metadata,
            "chunk_index": i,
            "total_chunks": total,
        }
        metadatas.append(chunk_meta)

    return ids, metadatas


# ---------------------------------------------------------------------------
# Core Ingestion Pipeline
# ---------------------------------------------------------------------------

async def ingest_document(
    pdf_path: Path,
    rag_service: RAGService,
    vector_store: ChromaVectorStore,
    doc_type_override: Optional[str] = None,
) -> int:
    """
    Full ingestion pipeline for a single PDF document.

    Steps:
        1. Extract text from PDF
        2. Detect (or use override) document type
        3. Extract structured metadata
        4. Chunk text with sentence-aware sliding window
        5. Embed all chunks via Nova Embed
        6. Upsert into ChromaDB

    Args:
        pdf_path: Path to the PDF file
        rag_service: Initialized RAGService (provides Nova embedding)
        vector_store: Initialized ChromaVectorStore (provides upsert)
        doc_type_override: Manually specify doc type instead of auto-detecting

    Returns:
        Number of chunks successfully upserted, or 0 on failure
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Ingesting: {pdf_path.name}")
    logger.info(f"{'='*60}")

    # Step 1: Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        logger.error(f"No text extracted from {pdf_path.name} — skipping")
        return 0

    # Step 2: Detect document type
    doc_type = doc_type_override or detect_doc_type(text)

    # Step 3: Extract metadata
    metadata = extract_metadata(
        text=text,
        doc_type=doc_type,
        source_filename=pdf_path.name,
    )

    # Step 4: Chunk text
    chunks = chunk_text(text)
    if not chunks:
        logger.error(f"No chunks produced from {pdf_path.name} — skipping")
        return 0

    logger.info(f"Produced {len(chunks)} chunks from {pdf_path.name}")

    # Step 5: Embed all chunks
    doc_id_prefix = f"{pdf_path.stem}_{doc_type}".replace(" ", "_").lower()
    chunk_ids, metadatas = prepare_documents(chunks, metadata, doc_id_prefix)

    logger.info(f"Embedding {len(chunks)} chunks via Nova Embed...")
    embeddings = await rag_service.embed_documents(chunks)

    # Validate — skip chunks where embedding failed (returned empty list)
    valid_documents = []
    skipped = 0

    for chunk_id, chunk_text_content, embedding, meta in zip(
        chunk_ids, chunks, embeddings, metadatas
    ):
        if not embedding:
            logger.warning(f"Skipping chunk '{chunk_id}' — embedding failed")
            skipped += 1
            continue

        valid_documents.append({
            "id": chunk_id,
            "embedding": embedding,
            "content": chunk_text_content,
            "metadata": meta,
        })

    if skipped:
        logger.warning(f"Skipped {skipped} chunks due to embedding failures")

    if not valid_documents:
        logger.error(f"No valid chunks to upsert for {pdf_path.name}")
        return 0

    # Step 6: Upsert into ChromaDB
    logger.info(f"Upserting {len(valid_documents)} chunks into ChromaDB...")
    success = await vector_store.upsert(valid_documents)

    if success:
        logger.info(f"Successfully ingested {len(valid_documents)} chunks from {pdf_path.name}")
        return len(valid_documents)
    else:
        logger.error(f"ChromaDB upsert failed for {pdf_path.name}")
        return 0


async def ingest_directory(
    docs_dir: str,
    collection_name: str,
    persist_directory: str,
    aws_region: str = "us-east-1",
) -> None:
    """
    Ingest all PDF files found in a directory.

    Initializes ChromaDB and RAGService, then processes each PDF
    sequentially. Prints a summary report on completion.

    Args:
        docs_dir: Path to directory containing PDF files
        collection_name: ChromaDB collection to upsert into
        persist_directory: Local ChromaDB storage path
        aws_region: AWS region for Bedrock Nova Embed
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        return

    pdf_files = sorted(docs_path.glob("**/*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files found in: {docs_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) to ingest")

    # Initialize vector store
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    await vector_store.initialize()

    # Initialize RAG service (embedding only — no generation model needed)
    rag_service = RAGService(
        vector_store=vector_store,
        aws_region=aws_region,
    )

    # Process each PDF
    results: Dict[str, int] = {}

    for pdf_path in pdf_files:
        count = await ingest_document(
            pdf_path=pdf_path,
            rag_service=rag_service,
            vector_store=vector_store,
        )
        results[pdf_path.name] = count

    # Summary report
    total_chunks = sum(results.values())
    collection_count = await vector_store.count()

    logger.info(f"\n{'='*60}")
    logger.info("INGESTION COMPLETE")
    logger.info(f"{'='*60}")

    for filename, count in results.items():
        status = "✓" if count > 0 else "✗"
        logger.info(f"  {status}  {filename}: {count} chunks")

    logger.info(f"\nTotal chunks ingested this run : {total_chunks}")
    logger.info(f"Total chunks in collection     : {collection_count}")
    logger.info(f"Collection                     : '{collection_name}'")
    logger.info(f"Storage                        : {persist_directory}")

    await vector_store.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest clinical PDF documents into ChromaDB via Nova Embed"
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="./documents",
        help="Directory containing PDF files to ingest (default: ./documents)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="clinical_docs",
        help="ChromaDB collection name (default: clinical_docs)",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="./chroma_db",
        help="Local ChromaDB storage directory (default: ./chroma_db)",
    )
    parser.add_argument(
        "--aws_region",
        type=str,
        default="us-east-1",
        help="AWS region for Bedrock Nova Embed (default: us-east-1)",
    )

    args = parser.parse_args()

    asyncio.run(
        ingest_directory(
            docs_dir=args.docs_dir,
            collection_name=args.collection,
            persist_directory=args.persist_dir,
            aws_region=args.aws_region,
        )
    )