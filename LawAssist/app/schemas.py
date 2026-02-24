from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SummarizeOptions(BaseModel):
    style: str = Field(default="executive", description="executive|bullet|detailed")
    bullets: bool = Field(default=True)
    max_words: Optional[int] = Field(default=None, description="Soft cap for summary length")
    include_actions: bool = Field(default=True)
    k_retrieval: int = Field(default=4, ge=0, le=12)
    tags: Optional[List[str]] = None

class SummarizeResponse(BaseModel):
    language: str
    summary: str
    key_points: Optional[List[str]] = None
    action_items: Optional[List[str]] = None
    meta: Dict[str, Any]

class IngestResponse(BaseModel):
    chunks_added: int
    doc_id: str
    meta: Dict[str, Any]