"""Central settings for the whole project (paths, models, retrieval knobs)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "bitext_raw.csv"
KB_CSV = DATA_DIR / "kb_rows.csv"            # rows used as the knowledge base
TEST_CSV = DATA_DIR / "test_rows.csv"        # held-out rows for evaluation
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"     # processed chunks (for BM25)
FAISS_DIR = DATA_DIR / "faiss_index"
FEEDBACK_JSONL = DATA_DIR / "feedback.jsonl" # user feedback + validator verdicts

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
ROWS_PER_INTENT_KB = 100   # 27 intents x 100 = 2,700 KB rows
ROWS_PER_INTENT_TEST = 3   # 27 intents x 3  = 81 evaluation queries
SEED = 42

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")          # "gemini" or "openai"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
LLM_FALLBACKS = [m for m in os.getenv("LLM_FALLBACKS", "gemini-3-flash-preview,gemini-2.5-flash-lite").split(",") if m]
LLM_REQUESTS_PER_MIN = int(os.getenv("LLM_REQUESTS_PER_MIN", "12"))  # stay under free-tier limits

TOP_K = 4               # final chunks passed to the LLM
CANDIDATES = 20         # candidates fetched by each retriever before fusion
SEMANTIC_WEIGHT = 0.6   # hybrid weights (semantic vs BM25)
BM25_WEIGHT = 0.4
MAX_RETRIES = 2         # feedback loop: re-query attempts if validation fails
K_STEP = 2              # how much k grows on each retry
