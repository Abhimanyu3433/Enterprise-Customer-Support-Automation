"""Retrieval module: top-k semantic search (FAISS), keyword search (BM25),
hybrid fusion (weighted Reciprocal Rank Fusion) and an optional cross-encoder re-ranker."""
import json
import re
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src import config
from src.ingestion.build_index import get_embeddings


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self, store: FAISS, chunks: list[Document], use_reranker: bool = True):
        self.store = store
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(c.page_content) for c in chunks])
        self.reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(config.RERANK_MODEL)

    # 1) Semantic top-k
    def semantic(self, query: str, k: int) -> list[Document]:
        return self.store.similarity_search(query, k=k)

    # 2) Keyword top-k
    def keyword(self, query: str, k: int) -> list[Document]:
        scores = self.bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top]

    # 3) Hybrid = weighted RRF of both ranked lists
    def hybrid(self, query: str, k: int) -> list[Document]:
        fused, docs = {}, {}
        for weight, ranked in [(config.SEMANTIC_WEIGHT, self.semantic(query, config.CANDIDATES)),
                               (config.BM25_WEIGHT, self.keyword(query, config.CANDIDATES))]:
            for rank, d in enumerate(ranked):
                cid = d.metadata["chunk_id"]
                fused[cid] = fused.get(cid, 0) + weight / (60 + rank)
                docs[cid] = d
        order = sorted(fused, key=fused.get, reverse=True)
        candidates = [docs[c] for c in order[: max(k * 3, config.CANDIDATES)]]
        return self.rerank(query, candidates, k)

    # 4) Optional re-ranker
    def rerank(self, query: str, docs: list[Document], k: int) -> list[Document]:
        if not self.reranker:
            return docs[:k]
        scores = self.reranker.predict([(query, d.page_content) for d in docs])
        return [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)][:k]


def load_chunks() -> list[Document]:
    with open(config.CHUNKS_JSONL) as f:
        return [Document(page_content=r["text"], metadata=r["metadata"]) for r in map(json.loads, f)]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    store = FAISS.load_local(str(config.FAISS_DIR), get_embeddings(), allow_dangerous_deserialization=True)
    return HybridRetriever(store, load_chunks())
