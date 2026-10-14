"""FastAPI backend for the agentic RAG customer-support assistant.

Run locally:  uvicorn api.main:app --reload --port 8000      (Swagger UI at http://localhost:8000/docs)
"""
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import FeedbackRequest, FeedbackResponse, HealthResponse, QueryRequest, QueryResponse
from src import config
from src.graph.workflow import run_agentic_rag
from src.retrieval.retriever import get_retriever
from src.tools.feedback import log_user_feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_retriever()  # load FAISS, BM25 and the re-ranker once at start-up, not on the first request
    yield


app = FastAPI(
    title="Customer Support Agentic RAG API",
    description="Multi-agent RAG (LangGraph) over the Bitext customer-support knowledge base.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", llm_provider=config.LLM_PROVIDER, llm_model=config.LLM_MODEL,
                          vectors_indexed=get_retriever().store.index.ntotal)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    try:
        out = run_agentic_rag(req.query, session_id)
    except Exception as e:  # LLM/provider errors -> clean 502 instead of a stack trace
        raise HTTPException(status_code=502, detail=f"Agent pipeline failed: {e}") from e
    return QueryResponse(session_id=session_id, answer=out["answer"], sources=out["sources"],
                         retrieved_context=out["retrieved_context"], agents_invoked=out["agents_invoked"],
                         confidence=max(0.0, min(1.0, float(out["confidence"]))))


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    log_user_feedback(req.session_id, req.query, req.answer, req.rating, req.correction)
    return FeedbackResponse()
