"""Data preparation: load -> clean -> split KB/test -> chunk (with overlap) -> embed -> FAISS.

Run:  python -m src.ingestion.build_index
"""
import json
import re

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)


def load_raw() -> pd.DataFrame:
    """Download the Bitext dataset once and cache it as CSV."""
    if not config.RAW_CSV.exists():
        from datasets import load_dataset
        config.DATA_DIR.mkdir(exist_ok=True)
        load_dataset(config.DATASET_NAME, split="train").to_pandas().to_csv(config.RAW_CSV, index=False)
    return pd.read_csv(config.RAW_CSV)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["instruction", "category", "intent", "response"]].dropna()
    for col in ["instruction", "response"]:
        df[col] = df[col].astype(str).map(lambda s: re.sub(r"\s+", " ", s).strip())
    df = df[(df.instruction.str.len() > 3) & (df.response.str.len() > 20)]
    df = df.drop_duplicates(subset=["instruction"]).reset_index(drop=True)
    df["row_id"] = df.index
    return df


def split_kb_test(df: pd.DataFrame):
    """Stratified per intent: a held-out test set never enters the knowledge base."""
    test = df.groupby("intent", group_keys=False).sample(config.ROWS_PER_INTENT_TEST, random_state=config.SEED)
    rest = df.drop(test.index)
    kb = rest.sample(frac=1, random_state=config.SEED).groupby("intent").head(config.ROWS_PER_INTENT_KB)
    return kb.reset_index(drop=True), test.reset_index(drop=True)


def feedback_corrections() -> list[dict]:
    """Feedback loop: user corrections become new 'verified' KB entries on rebuild."""
    rows = []
    if config.FEEDBACK_JSONL.exists():
        for i, line in enumerate(config.FEEDBACK_JSONL.read_text().splitlines()):
            fb = json.loads(line)
            if fb.get("type") == "user" and fb.get("correction"):
                rows.append({"question": fb["query"], "answer": fb["correction"], "source": f"user_correction#{i}",
                             "intent": fb.get("intent") or "unknown", "category": fb.get("category") or "unknown"})
    return rows


def to_chunks(kb: pd.DataFrame) -> list[Document]:
    """Split each support ANSWER into overlapping chunks and prefix every chunk with its customer question,
    so no chunk is left without an answer and every chunk stays searchable by the question."""
    rows = [{"question": r.instruction, "answer": r.response, "source": f"bitext_faq#{r.row_id}",
             "intent": r.intent, "category": r.category} for r in kb.itertuples()] + feedback_corrections()
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = []
    for r in rows:
        for piece in splitter.split_text(r["answer"]):
            chunks.append(Document(
                page_content=f"Customer question: {r['question']}\nSupport answer: {piece}",
                metadata={"source": r["source"], "intent": r["intent"], "category": r["category"],
                          "chunk_id": len(chunks)},
            ))
    return chunks


def main():
    df = clean(load_raw())
    kb, test = split_kb_test(df)
    kb.to_csv(config.KB_CSV, index=False)
    test.to_csv(config.TEST_CSV, index=False)

    chunks = to_chunks(kb)
    with open(config.CHUNKS_JSONL, "w") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.page_content, "metadata": c.metadata}) + "\n")

    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(config.FAISS_DIR))
    print(f"Clean rows: {len(df)} | KB rows: {len(kb)} | Test rows: {len(test)}")
    print(f"Chunks: {len(chunks)} | FAISS vectors: {store.index.ntotal} -> {config.FAISS_DIR}")


if __name__ == "__main__":
    main()
