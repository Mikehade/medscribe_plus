from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="The query string to search for")
    top_k: int = Field(default=5, description="Number of document chunks to retrieve")
    specialty: Optional[str] = Field(default=None, description="Optional medical specialty to filter results")
    doc_type: Optional[str] = Field(default=None, description="Optional document type to filter results")
    score_threshold: Optional[float] = Field(default=0.40, description="Minimum similarity score")

class RetrievedChunkSchema(BaseModel):
    chunk_id: str
    content: str
    score: float
    source: str
    doc_type: str
    metadata: Dict[str, Any]

class RAGQueryResponse(BaseModel):
    success: bool
    query: str
    results_found: int
    chunks: List[RetrievedChunkSchema]
    message: str

class IngestResponse(BaseModel):
    success: bool
    filename: str
    chunks_ingested: int
    message: str
