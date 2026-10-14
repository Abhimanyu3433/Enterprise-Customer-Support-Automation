"""Retriever and ingestion tests: BM25, semantic, hybrid fusion and chunking."""
import pandas as pd

from src import config
from src.ingestion.build_index import clean, split_kb_test, to_chunks


def test_keyword_search_finds_exact_terms(retriever):
    top = retriever.keyword("refund", k=1)[0]
    assert top.metadata["intent"] == "track_refund"


def test_semantic_returns_k_documents(retriever):
    assert len(retriever.semantic("cancel order", k=3)) == 3


def test_hybrid_returns_unique_top_k_with_metadata(retriever):
    docs = retriever.hybrid("payment methods UPI", k=3)
    ids = [d.metadata["chunk_id"] for d in docs]
    assert len(docs) == 3 and len(set(ids)) == 3
    assert all({"source", "intent", "category"} <= d.metadata.keys() for d in docs)


def test_hybrid_fusion_uses_keyword_scores(monkeypatch, retriever):
    # fake embeddings carry no meaning, so weight BM25 fully to check the fusion ranks keyword hits first
    monkeypatch.setattr(config, "SEMANTIC_WEIGHT", 0.0)
    monkeypatch.setattr(config, "BM25_WEIGHT", 1.0)
    assert retriever.hybrid("payment methods UPI", k=1)[0].metadata["intent"] == "check_payment_methods"


def test_clean_drops_duplicates_and_short_rows():
    df = pd.DataFrame({"instruction": ["cancel order", "cancel order", "hi"],
                       "category": ["ORDER"] * 3, "intent": ["cancel_order"] * 3,
                       "response": ["x" * 50, "y" * 50, "z" * 50]})
    assert len(clean(df)) == 1


def test_split_keeps_test_rows_out_of_kb():
    df = pd.DataFrame({"instruction": [f"question {i}" for i in range(40)], "category": "C",
                       "intent": [f"i{i % 2}" for i in range(40)], "response": "r" * 30})
    kb, test = split_kb_test(clean(df))
    assert set(kb.instruction).isdisjoint(test.instruction)
    assert test.intent.value_counts().min() == 3


def test_every_chunk_keeps_its_question():
    kb = pd.DataFrame([{"instruction": "how to cancel", "category": "ORDER", "intent": "cancel_order",
                        "response": "Step. " * 300, "row_id": 7}])
    chunks = to_chunks(kb)
    assert len(chunks) > 1
    assert all(c.page_content.startswith("Customer question: how to cancel\nSupport answer:") for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
