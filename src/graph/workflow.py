"""LangGraph multi-agent workflow (planner-executor) with memory, validation and a retry feedback loop.

    START -> planner --(plan)--> retrieve -> validate --fail & retries left--> retrieve (rewritten query, larger k)
                                              |pass / retries exhausted
                                              v
                                           respond -> [escalate] -> END
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src import config
from src.agents.agents import escalation_agent, planner_agent, validator_agent
from src.tools.feedback import log_validator_verdict
from src.tools.rag_tool import generate_answer, rag_search

ESCALATION_NOTE = "I've escalated this to our human support team; an agent will contact you shortly."


class SupportState(TypedDict, total=False):
    messages: Annotated[list, add_messages]   # conversational memory (persisted per session)
    query: str
    history: str
    understanding: dict
    plan: list[str]
    step_idx: int
    search_query: str
    k: int
    attempt: int
    retrieved: dict
    verdict: dict
    answer: str
    sources: list[str]
    confidence: float
    escalation: dict
    trace: list[str]                           # agents invoked this turn (agent trace)


def _history(messages, last_n=6) -> str:
    return "\n".join(f"{'Customer' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                     for m in messages[-last_n:])


# ---------------- Nodes ----------------
def planner_node(state: SupportState):
    query = state["messages"][-1].content
    history = _history(state["messages"][:-1])
    out = planner_agent(query, history)
    return {"query": query, "history": history, "understanding": out["understanding"], "plan": out["plan"],
            "step_idx": 0, "search_query": out["understanding"]["standalone_query"], "k": config.TOP_K,
            "attempt": 0, "escalation": None, "trace": ["planner", "query_understanding"]}


def retrieve_node(state: SupportState):  # RAG Retrieval Agent -> uses the RAG tool
    retrieved = rag_search.invoke({"query": state["search_query"], "k": state["k"]})
    return {"retrieved": retrieved, "step_idx": state["step_idx"] + 1, "trace": state["trace"] + ["rag_retrieval"]}


def validate_node(state: SupportState):
    verdict = validator_agent(state["search_query"], state["understanding"]["intent"],
                              state["retrieved"]["context"])
    log_validator_verdict(state["query"], state["attempt"], state["k"], verdict)
    update = {"verdict": verdict, "trace": state["trace"] + ["validator"]}
    if not verdict["is_grounded"] and state["attempt"] < config.MAX_RETRIES:
        # Feedback loop: re-query with a rewritten query and a larger k
        return {**update, "search_query": verdict["better_query"] or state["search_query"],
                "k": state["k"] + config.K_STEP, "attempt": state["attempt"] + 1,
                "step_idx": state["plan"].index("retrieve")}
    return {**update, "step_idx": state["step_idx"] + 1}


def respond_node(state: SupportState):  # generation, chained on RAG output
    # Answer the customer's own words (with history); the rewritten query is only used for searching
    out = generate_answer(state["query"], state["retrieved"], state["history"])
    confidence = round(min(out.confidence, state["verdict"]["score"]), 2)
    return {"answer": out.answer, "sources": out.sources or state["retrieved"]["sources"],
            "confidence": confidence, "messages": [AIMessage(content=out.answer)],
            "step_idx": state["step_idx"] + 1, "trace": state["trace"] + ["responder"]}


def escalate_node(state: SupportState):
    esc = escalation_agent(state["query"], state["answer"], state["verdict"]["is_grounded"],
                           state["understanding"]["intent"])
    update = {"escalation": esc, "step_idx": state["step_idx"] + 1,
              "trace": state["trace"] + ["escalation", "sentiment_analysis"]}
    if esc["escalate"]:
        note = "\n\n" + ESCALATION_NOTE
        update["answer"] = state["answer"] + note
        update["messages"] = [AIMessage(content=note.strip())]
    return update


# ---------------- Executor routing (follows the planner's plan) ----------------
def next_step(state: SupportState) -> str:
    i = state["step_idx"]
    return state["plan"][i] if i < len(state["plan"]) else END


# Only the transitions that can actually happen (keeps the flow diagram readable)
ROUTES = {
    "planner": {"retrieve": "retrieve"},
    "retrieve": {"validate": "validate"},
    "validate": {"retrieve": "retrieve", "respond": "respond"},   # retrieve = feedback-loop retry
    "respond": {"escalate": "escalate", END: END},
    "escalate": {END: END},
}


def build_graph():
    g = StateGraph(SupportState)
    g.add_node("planner", planner_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("validate", validate_node)
    g.add_node("respond", respond_node)
    g.add_node("escalate", escalate_node)
    g.add_edge(START, "planner")
    for node, targets in ROUTES.items():
        g.add_conditional_edges(node, next_step, targets)
    return g.compile(checkpointer=MemorySaver())


graph = build_graph()


def run_agentic_rag(query: str, session_id: str = "default") -> dict:
    """Single entry point (re-used by evaluation now and FastAPI in Section 2)."""
    s = graph.invoke({"messages": [HumanMessage(content=query)]},
                     config={"configurable": {"thread_id": session_id}})
    return {"answer": s["answer"], "sources": s["sources"], "retrieved_context": s["retrieved"]["context"],
            "agents_invoked": s["trace"], "confidence": s["confidence"],
            "intent": s["understanding"]["intent"], "escalation": s.get("escalation"),
            "retrieved_intents": s["retrieved"]["intents"]}


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())          # LangGraph flow diagram (paste into mermaid.live)
    print(run_agentic_rag("I want to cancel order 12345, how do I do it?", "demo"))
    print(run_agentic_rag("and will I be charged a fee for that?", "demo"))   # uses memory
