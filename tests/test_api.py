"""API tests with FastAPI's TestClient. The agent graph and the vector store are faked."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.main as main

FAKE_RESULT = {"answer": "Click Cancel.", "sources": ["bitext_faq#0"], "retrieved_context": ["ctx"],
               "agents_invoked": ["planner", "rag_retrieval", "validator", "responder"], "confidence": 0.87}


@pytest.fixture
def client(monkeypatch):
    fake_retriever = SimpleNamespace(store=SimpleNamespace(index=SimpleNamespace(ntotal=5)))
    monkeypatch.setattr(main, "get_retriever", lambda: fake_retriever)
    monkeypatch.setattr(main, "run_agentic_rag", lambda q, s: FAKE_RESULT)
    monkeypatch.setattr(main, "log_user_feedback", lambda *a, **k: None)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["vectors_indexed"] == 5


def test_query_returns_answer_context_sources_and_trace(client):
    r = client.post("/query", json={"session_id": "abc123", "query": "How do I cancel?"})
    body = r.json()
    assert r.status_code == 200
    assert body["session_id"] == "abc123" and body["answer"] == "Click Cancel."
    assert body["sources"] and body["retrieved_context"] and body["agents_invoked"][0] == "planner"
    assert 0 <= body["confidence"] <= 1


def test_query_creates_session_id_when_missing(client):
    assert client.post("/query", json={"query": "hi"}).json()["session_id"]


def test_query_rejects_empty_question(client):
    assert client.post("/query", json={"query": ""}).status_code == 422


def test_feedback(client):
    r = client.post("/feedback", json={"session_id": "s", "query": "q", "answer": "a", "rating": "up"})
    assert r.status_code == 200 and r.json()["status"] == "stored"


def test_feedback_rejects_bad_rating(client):
    r = client.post("/feedback", json={"session_id": "s", "query": "q", "answer": "a", "rating": "meh"})
    assert r.status_code == 422
