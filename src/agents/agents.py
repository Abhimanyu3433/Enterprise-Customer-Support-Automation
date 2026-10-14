"""Specialised agents for Use Case C (Enterprise Customer Support Automation).

Agents as Tools:
  * Query Understanding Agent is exposed as a tool and invoked by the Planner.
  * Sentiment Analysis Agent is exposed as a tool and invoked by the Escalation Agent.
  * RAG (src/tools/rag_tool.py) is the shared tool used by the RAG Retrieval Agent.
"""
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.tools.rag_tool import get_llm

INTENTS = ["cancel_order", "change_order", "place_order", "check_cancellation_fee", "delivery_options",
           "delivery_period", "complaint", "review", "check_invoice", "get_invoice", "newsletter_subscription",
           "check_payment_methods", "payment_issue", "check_refund_policy", "get_refund", "track_refund",
           "change_shipping_address", "set_up_shipping_address", "create_account", "delete_account",
           "edit_account", "switch_account", "recover_password", "registration_problems", "track_order",
           "contact_customer_service", "contact_human_agent"]
CATEGORIES = ["ACCOUNT", "CANCELLATION_FEE", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE", "NEWSLETTER",
              "ORDER", "PAYMENT", "REFUND", "SHIPPING_ADDRESS"]
STEPS = ["retrieve", "validate", "respond", "escalate"]


# ---------------- Query Understanding Agent ----------------
class QueryUnderstanding(BaseModel):
    intent: str = Field(description=f"One of: {', '.join(INTENTS)}")
    category: str = Field(description=f"One of: {', '.join(CATEGORIES)}")
    standalone_query: str = Field(description="The question rewritten to be self-contained using the history")


@tool
def query_understanding_agent(query: str, history: str = "") -> dict:
    """Identify the customer's intent and category, and rewrite follow-ups into a standalone query."""
    prompt = (f"Classify the customer message.\nConversation history:\n{history or '(none)'}\n\n"
              f"Customer message: {query}")
    return get_llm().with_structured_output(QueryUnderstanding).invoke(prompt).model_dump()


# ---------------- Sentiment Analysis Agent ----------------
class Sentiment(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    urgency: Literal["low", "medium", "high"]
    satisfaction_score: float = Field(description="0 (very unhappy) to 1 (very satisfied)")


@tool
def sentiment_agent(text: str) -> dict:
    """Evaluate customer satisfaction and urgency of a message."""
    return get_llm().with_structured_output(Sentiment).invoke(
        f"Assess sentiment, urgency and satisfaction of this customer message:\n{text}").model_dump()


# ---------------- Escalation Agent (calls Sentiment Agent as a tool) ----------------
class EscalationDecision(BaseModel):
    escalate: bool
    reason: str


def escalation_agent(query: str, answer: str, is_grounded: bool, intent: str) -> dict:
    """Detect unresolved issues and route them to human support."""
    sentiment = sentiment_agent.invoke({"text": query})          # Agent-as-Tool call
    prompt = (
        "Decide if this support conversation must be escalated to a human agent. Escalate if the answer "
        "is not grounded, the customer explicitly asks for a human, or sentiment is negative with high urgency.\n"
        f"Intent: {intent}\nSentiment analysis: {sentiment}\nAnswer grounded: {is_grounded}\n"
        f"Customer: {query}\nBot answer: {answer}"
    )
    decision = get_llm().with_structured_output(EscalationDecision).invoke(prompt).model_dump()
    return {**decision, "sentiment": sentiment}


# ---------------- Planner (calls Query Understanding Agent as a tool) ----------------
class Plan(BaseModel):
    steps: list[Literal["retrieve", "validate", "respond", "escalate"]]
    rationale: str


def planner_agent(query: str, history: str) -> dict:
    understanding = query_understanding_agent.invoke({"query": query, "history": history})  # Agent-as-Tool
    prompt = (
        f"You plan how to handle a support request. Available steps (executed in order): {STEPS}.\n"
        "'retrieve', 'validate' and 'respond' are always required, in that order. Append 'escalate' "
        "when the message is a complaint, payment problem, asks for a human, or sounds angry/urgent.\n"
        f"Understanding: {understanding}\nCustomer message: {query}"
    )
    plan = get_llm().with_structured_output(Plan).invoke(prompt)
    steps = ["retrieve", "validate", "respond"] + (["escalate"] if "escalate" in plan.steps else [])
    return {"understanding": understanding, "plan": steps, "rationale": plan.rationale}


# ---------------- Validator Agent (retrieval relevance + grounding check) ----------------
class Verdict(BaseModel):
    is_relevant: bool = Field(description="Retrieved context is about the customer's question")
    is_grounded: bool = Field(description="Context contains enough information to answer")
    score: float = Field(description="0-1 relevance score")
    better_query: str = Field(description="A rewritten search query to use if validation fails")


def validator_agent(query: str, intent: str, context: list[str]) -> dict:
    prompt = ("Check whether the retrieved context is relevant and sufficient to answer the question. "
              "Placeholders like {{Order Number}} are normal and do not make the context insufficient.\n"
              f"Question: {query}\nDetected intent: {intent}\nContext:\n" + "\n---\n".join(context))
    v = get_llm().with_structured_output(Verdict).invoke(prompt).model_dump()
    # Pass when the context is relevant and either judged sufficient or scored >= 0.6
    v["is_grounded"] = v["is_relevant"] and (v["is_grounded"] or v["score"] >= 0.6)
    return v
