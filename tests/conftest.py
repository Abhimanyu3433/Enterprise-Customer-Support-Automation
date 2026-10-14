"""Shared test fixtures. Tests never call a real LLM or download models: everything external is faked."""
import os

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")

from src.retrieval.retriever import HybridRetriever  # noqa: E402

KB = [
    ("how do I cancel my order", "cancel_order", "Go to Orders, pick the order and click Cancel."),
    ("where is my refund", "track_refund", "Refunds reach your account within 5-7 business days."),
    ("what payment methods do you accept", "check_payment_methods", "We accept cards, PayPal and UPI."),
    ("I want to talk to a human agent", "contact_human_agent", "Call support or use live chat."),
    ("change my shipping address", "change_shipping_address", "Edit the address under Settings > Addresses."),
]


@pytest.fixture
def chunks():
    return [Document(page_content=f"Customer question: {q}\nSupport answer: {a}",
                     metadata={"source": f"bitext_faq#{i}", "intent": intent, "category": "TEST", "chunk_id": i})
            for i, (q, intent, a) in enumerate(KB)]


@pytest.fixture
def retriever(chunks):
    store = FAISS.from_documents(chunks, DeterministicFakeEmbedding(size=32))
    return HybridRetriever(store, chunks, use_reranker=False)


class FakeLLM:
    """Stands in for get_llm(): with_structured_output(schema).invoke(...) returns schema(**values[schema name])."""

    def __init__(self, values: dict):
        self.values = values

    def with_structured_output(self, schema):
        values = self.values[schema.__name__]

        class Runner:
            def invoke(self, _prompt):
                return schema(**values)
        return Runner()


@pytest.fixture
def fake_llm():
    return FakeLLM
