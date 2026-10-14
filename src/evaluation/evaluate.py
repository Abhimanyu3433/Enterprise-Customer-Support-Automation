"""Evaluation: Baseline RAG vs Agentic RAG on the held-out test set.

Metrics
  * BLEU               (sacrebleu, generated answer vs reference response)
  * ROUGE-1/2/L F1     (rouge-score)
  * Retrieval relevance = precision@k of retrieved chunks whose intent matches the true intent
  * Answer relevance    = cosine similarity between query and answer embeddings

Run:  python -m src.evaluation.evaluate            (optionally --n 27 for a quick run)
"""
import argparse
import uuid

import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

from src import config
from src.graph.workflow import ESCALATION_NOTE, run_agentic_rag
from src.ingestion.build_index import get_embeddings
from src.retrieval.retriever import get_retriever
from src.tools.rag_tool import format_docs, generate_answer

ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def baseline_rag(query: str) -> dict:
    """Plain RAG: semantic top-k only, single generation call, no agents/validation/retries."""
    retrieved = format_docs(get_retriever().semantic(query, config.TOP_K))
    out = generate_answer(query, retrieved)
    return {"answer": out.answer, "retrieved_intents": retrieved["intents"]}


def agentic_rag(query: str) -> dict:
    out = run_agentic_rag(query, session_id=str(uuid.uuid4()))  # fresh memory per test query
    out["answer"] = out["answer"].replace(ESCALATION_NOTE, "").strip()  # score the answer itself, not the hand-off note
    return out


def score(rows: list[dict], emb) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["bleu"] = [sacrebleu.sentence_bleu(a, [r]).score for a, r in zip(df.answer, df.reference)]
    for m in ["rouge1", "rouge2", "rougeL"]:
        df[m] = [ROUGE.score(r, a)[m].fmeasure for a, r in zip(df.answer, df.reference)]
    df["retrieval_relevance"] = [np.mean([i == t for i in ri]) if ri else 0.0
                                 for ri, t in zip(df.retrieved_intents, df.intent)]
    q, a = np.array(emb.embed_documents(list(df["query"]))), np.array(emb.embed_documents(list(df.answer)))
    df["answer_relevance"] = (q * a).sum(1) / (np.linalg.norm(q, axis=1) * np.linalg.norm(a, axis=1))
    return df


def main(n: int | None):
    test = pd.read_csv(config.TEST_CSV)
    if n:
        test = test.groupby("intent", group_keys=False).head(max(1, n // test.intent.nunique()))
    emb = get_embeddings()
    results = {}
    for name, fn in [("baseline_rag", baseline_rag), ("agentic_rag", agentic_rag)]:
        rows = []
        for r in test.itertuples():
            out = fn(r.instruction)
            rows.append({"query": r.instruction, "intent": r.intent, "reference": r.response,
                         "answer": out["answer"], "retrieved_intents": out["retrieved_intents"]})
            print(f"[{name}] {len(rows)}/{len(test)}", end="\r")
        df = score(rows, emb)
        df.to_csv(config.DATA_DIR / f"eval_{name}.csv", index=False)
        results[name] = df[["bleu", "rouge1", "rouge2", "rougeL", "retrieval_relevance", "answer_relevance"]].mean()

    table = pd.DataFrame(results).T.round(4)
    table.to_csv(config.DATA_DIR / "eval_summary.csv")
    print("\n\nBaseline RAG vs Agentic RAG\n", table.to_markdown())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="approx. number of test queries (default: all 81)")
    main(ap.parse_args().n)
