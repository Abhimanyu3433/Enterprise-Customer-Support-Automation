"""RAG exposed as reusable tools + the generation module (structured JSON output)."""
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src import config
from src.retrieval.retriever import get_retriever


class LLMWithFallbacks:
    """Primary model + backup models: if one is overloaded (e.g. 503) or rate-limited, the next is tried."""

    def __init__(self, models):
        self.models = models

    def with_structured_output(self, schema):
        runs = [m.with_structured_output(schema) for m in self.models]
        return runs[0].with_fallbacks(runs[1:]) if len(runs) > 1 else runs[0]

    def invoke(self, *args, **kwargs):
        return self.models[0].with_fallbacks(self.models[1:]).invoke(*args, **kwargs)


def _make_model(name: str, limiter):
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, temperature=0, rate_limiter=limiter, max_retries=3)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=name, temperature=0, rate_limiter=limiter, max_retries=3)


@lru_cache(maxsize=1)
def get_llm() -> LLMWithFallbacks:
    """LLM used by every agent. Provider/model/fallbacks are chosen in .env."""
    limiter = InMemoryRateLimiter(requests_per_second=config.LLM_REQUESTS_PER_MIN / 60)
    names = [config.LLM_MODEL] + (config.LLM_FALLBACKS if config.LLM_PROVIDER == "gemini" else [])
    return LLMWithFallbacks([_make_model(n, limiter) for n in names])


# ---------- Retrieval tool (callable by any agent while reasoning) ----------
def format_docs(docs: list[Document]) -> dict:
    return {
        "context": [d.page_content for d in docs],
        "sources": [d.metadata["source"] for d in docs],
        "intents": [d.metadata["intent"] for d in docs],
    }


@tool
def rag_search(query: str, k: int = config.TOP_K) -> dict:
    """Search the customer-support knowledge base (hybrid FAISS + BM25 + re-ranker).
    Returns retrieved context passages, their source ids and intents."""
    return format_docs(get_retriever().hybrid(query, k))


# ---------- Generation module ----------
class RAGAnswer(BaseModel):
    answer: str = Field(description="Answer to the customer, based only on the context")
    sources: list[str] = Field(description="Source ids from the context that support the answer")
    confidence: float = Field(description="0-1 confidence that the answer is fully supported by the context")


GEN_PROMPT = """You are a customer-support assistant. Answer the customer's question using ONLY the context.
Keep placeholders such as {{{{Order Number}}}} exactly as written. If the context does not contain the
answer, say so and set confidence low.

Conversation so far:
{history}

Context:
{context}

Customer question: {question}"""


def generate_answer(question: str, retrieved: dict, history: str = "") -> RAGAnswer:
    context = "\n\n".join(f"[{s}] {c}" for s, c in zip(retrieved["sources"], retrieved["context"]))
    llm = get_llm().with_structured_output(RAGAnswer)
    return llm.invoke(GEN_PROMPT.format(history=history or "(none)", context=context, question=question))


@tool
def rag_answer(question: str) -> dict:
    """Full RAG tool: retrieve from the knowledge base and generate a grounded JSON answer."""
    retrieved = rag_search.invoke({"query": question})
    out = generate_answer(question, retrieved)
    return {**out.model_dump(), "retrieved_context": retrieved["context"]}
