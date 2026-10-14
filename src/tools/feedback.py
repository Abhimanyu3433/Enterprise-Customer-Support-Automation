"""Feedback store: user thumbs up/down + corrections, and validator verdicts.
Stored as JSONL for later tuning (corrections are re-ingested by build_index.py)."""
import json
from datetime import datetime, timezone

from src import config


def log_feedback(record: dict) -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with open(config.FEEDBACK_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_user_feedback(session_id: str, query: str, answer: str, rating: str,
                      correction: str | None = None, intent: str | None = None) -> None:
    """rating: 'up' or 'down'. A correction (optional) becomes a new KB document on re-index."""
    log_feedback({"type": "user", "session_id": session_id, "query": query, "answer": answer,
                  "rating": rating, "correction": correction, "intent": intent})


def log_validator_verdict(query: str, attempt: int, k: int, verdict: dict) -> None:
    log_feedback({"type": "validator", "query": query, "attempt": attempt, "k": k, **verdict})


def feedback_summary() -> dict:
    """Quick stats used for tuning decisions (e.g. which intents get most thumbs-down)."""
    if not config.FEEDBACK_JSONL.exists():
        return {}
    rows = [json.loads(line) for line in config.FEEDBACK_JSONL.read_text().splitlines()]
    user = [r for r in rows if r["type"] == "user"]
    val = [r for r in rows if r["type"] == "validator"]
    return {
        "thumbs_up": sum(r["rating"] == "up" for r in user),
        "thumbs_down": sum(r["rating"] == "down" for r in user),
        "corrections": sum(bool(r.get("correction")) for r in user),
        "validator_fail_rate": round(sum(not r["is_grounded"] for r in val) / len(val), 3) if val else None,
    }
