"""Smoke test for a running API: checks /health, /query and /feedback. Exit code 0 = all good.

Usage:  python -m api.smoke_test [base_url]        (default http://localhost:8000)
Set SMOKE_SKIP_LLM=1 to skip /query (e.g. in CI when no LLM API key is configured).
"""
import json
import os
import sys
import time

import requests


def main(base: str) -> int:
    for _ in range(90):                                   # wait up to ~3 min for start-up
        try:
            h = requests.get(f"{base}/health", timeout=3)
            break
        except requests.ConnectionError:
            time.sleep(2)
    else:
        print("API did not start")
        return 1
    print("GET /health ->", h.status_code, h.json())

    body = {"session_id": "smoke-test", "query": "How can I cancel my order 12345?"}
    answer, query_ok = "", True
    if os.getenv("SMOKE_SKIP_LLM") == "1":
        print("POST /query -> skipped (SMOKE_SKIP_LLM=1)")
    else:
        q = requests.post(f"{base}/query", json=body, timeout=180)
        print("POST /query ->", q.status_code)
        print(json.dumps(q.json(), indent=2)[:3000])
        answer = q.json().get("answer", "")
        query_ok = q.status_code == 200 and bool(answer)

    f = requests.post(f"{base}/feedback", timeout=30, json={
        "session_id": "smoke-test", "query": body["query"], "answer": answer, "rating": "up"})
    print("POST /feedback ->", f.status_code, f.json())

    ok = h.status_code == 200 and h.json().get("vectors_indexed", 0) > 0 and query_ok and f.status_code == 200
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
