"""Pydantic request/response schemas for the FastAPI service."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Conversation id; reuse it to keep chat memory")
    query: str = Field(..., min_length=1, max_length=2000, description="Customer question")

    model_config = {"json_schema_extra": {"examples": [
        {"session_id": "abc123", "query": "How can I cancel my order 12345?"}]}}


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str]
    retrieved_context: list[str]
    agents_invoked: list[str]
    confidence: float = Field(..., ge=0, le=1)


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    answer: str
    rating: Literal["up", "down"]
    correction: Optional[str] = Field(None, description="Corrected answer (added to the KB on the next re-index)")


class FeedbackResponse(BaseModel):
    status: str = "stored"


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_model: str
    vectors_indexed: int
