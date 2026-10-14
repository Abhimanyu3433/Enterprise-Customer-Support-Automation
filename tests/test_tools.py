"""Tool tests: RAG tool, generation module, validator agent and feedback store (LLM is faked)."""
import json

from src import config
from src.agents import agents
from src.tools import feedback, rag_tool


def test_rag_search_tool_returns_context_and_sources(monkeypatch, retriever):
    monkeypatch.setattr(rag_tool, "get_retriever", lambda: retriever)
    out = rag_tool.rag_search.invoke({"query": "where is my refund", "k": 2})
    assert len(out["context"]) == len(out["sources"]) == len(out["intents"]) == 2
    assert all(s.startswith("bitext_faq#") for s in out["sources"])


def test_generate_answer_returns_structured_json(monkeypatch, fake_llm):
    monkeypatch.setattr(rag_tool, "get_llm", lambda: fake_llm(
        {"RAGAnswer": {"answer": "Click Cancel.", "sources": ["bitext_faq#0"], "confidence": 0.9}}))
    out = rag_tool.generate_answer("cancel?", {"context": ["c"], "sources": ["bitext_faq#0"]})
    assert out.answer == "Click Cancel." and out.sources == ["bitext_faq#0"] and 0 <= out.confidence <= 1


def test_validator_passes_relevant_context(monkeypatch, fake_llm):
    monkeypatch.setattr(agents, "get_llm", lambda: fake_llm({"Verdict": {
        "is_relevant": True, "is_grounded": False, "score": 0.8, "better_query": "x"}}))
    assert agents.validator_agent("q", "cancel_order", ["ctx"])["is_grounded"] is True


def test_validator_fails_irrelevant_context(monkeypatch, fake_llm):
    monkeypatch.setattr(agents, "get_llm", lambda: fake_llm({"Verdict": {
        "is_relevant": False, "is_grounded": True, "score": 0.9, "better_query": "better"}}))
    v = agents.validator_agent("q", "cancel_order", ["ctx"])
    assert v["is_grounded"] is False and v["better_query"] == "better"


def test_feedback_is_stored_and_summarised(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "FEEDBACK_JSONL", tmp_path / "feedback.jsonl")
    feedback.log_user_feedback("s1", "q", "a", "down", correction="better answer")
    feedback.log_validator_verdict("q", 0, 4, {"is_grounded": False, "score": 0.2})
    rows = [json.loads(line) for line in (tmp_path / "feedback.jsonl").read_text().splitlines()]
    assert rows[0]["correction"] == "better answer"
    s = feedback.feedback_summary()
    assert s["thumbs_down"] == 1 and s["corrections"] == 1 and s["validator_fail_rate"] == 1.0
