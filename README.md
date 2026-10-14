# Capstone Project 2 – Multi-Agent RAG for Enterprise Customer Support Automation

**Use case chosen:** C – Enterprise Customer Support Automation
**Stack:** Python · LangGraph · FAISS · BM25 · HuggingFace embeddings + cross-encoder · OpenAI `gpt-4o-mini` · FastAPI · Streamlit · Docker · GitHub Actions

---

## 1. Dataset and justification

**Bitext Customer Support LLM Chatbot Training Dataset** – `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (HuggingFace, licence CDLA-Sharing-1.0)

| Property | Value |
|---|---|
| Rows | 26,872 customer question → support answer pairs |
| Columns | `flags`, `instruction`, `category`, `intent`, `response` |
| Labels | 27 intents across 11 categories (ORDER, REFUND, PAYMENT, ACCOUNT, DELIVERY, …) |

Why this dataset fits the brief:

1. **Named in the PDF.** Section 3 suggests "Bitext or similar customer-support datasets" for this use case.
2. **It is a real knowledge base.** Each `response` is a policy-style FAQ answer, so it can be chunked, embedded and retrieved exactly as Section 1 requires.
3. **Every agent has something to work with.**
   - `intent` / `category` labels drive the **Query Understanding Agent**.
   - Colloquial, typo-ridden and offensive phrasings (flags `Q`, `Z`, `W`) exercise the **Sentiment** and **Escalation** agents.
   - The `contact_human_agent` and `complaint` intents give the Escalation Agent genuine routing cases.
4. **Reference answers allow proper evaluation.** Every question has a ground-truth `response`, so BLEU and ROUGE can be computed directly. The intent labels also give a retrieval-relevance score without manual labelling.
5. **Public and anonymised.** It is synthetic, contains no personal data, and uses placeholders such as `{{Order Number}}`.

**How it is used:**
- **Knowledge base:** 100 rows per intent (2,700 rows), which keeps embedding fast on a CPU.
- **Test set:** a separate 3 rows per intent (81 queries), held out so it never enters the index.

---

## 2. Folder structure

```
capstone-rag-agents/
├── data/                         # raw CSV, KB/test splits, chunks, FAISS index, feedback, eval results (generated)
├── notebooks/
│   └── section1_experiments.ipynb   # experiments, agent traces, evaluation (screenshots for report)
├── src/
│   ├── config.py                 # paths, models, k, weights, retry limit
│   ├── ingestion/build_index.py  # load → clean → split → chunk (overlap) → embed → FAISS
│   ├── retrieval/retriever.py    # semantic top-k, BM25, hybrid fusion, cross-encoder re-ranker
│   ├── tools/rag_tool.py         # RAG as a tool + generation module (JSON: answer, sources, confidence)
│   ├── tools/feedback.py         # thumbs up/down, corrections, validator verdicts (JSONL)
│   ├── agents/agents.py          # planner, query-understanding, validator, sentiment, escalation agents
│   ├── graph/workflow.py         # LangGraph state, graph, memory, retry loop
│   └── evaluation/evaluate.py    # BLEU, ROUGE, relevance; baseline vs agentic
├── api/                          # FastAPI app, Pydantic schemas, smoke test, Dockerfile
├── ui/                           # Streamlit main.py + Dockerfile
├── docker-compose.yml            # runs API + UI containers together
├── tests/                        # pytest: retriever, tools, API
├── .github/workflows/ci-cd.yml   # CI/CD: lint, test, build, push, deploy, validate, rollback
├── requirements.txt
├── .env.example
└── README.md
```

---

## 3. Section 1 – RAG pipeline and agentic workflow ✅ (built)

| PDF task | Implementation |
|---|---|
| **Data preparation** | `build_index.py`<br>• Cleans whitespace, drops empty and duplicate rows<br>• Stratified KB/test split<br>• `RecursiveCharacterTextSplitter` (500 characters, 80 overlap)<br>• `HuggingFaceEmbeddings` (all-MiniLM-L6-v2)<br>• FAISS store with metadata `source`, `intent`, `category`, `chunk_id` |
| **Retrieval module** | `retriever.py`<br>• Top-k semantic search (FAISS)<br>• Keyword search (BM25)<br>• Hybrid fusion: weighted Reciprocal Rank Fusion, 0.6 semantic / 0.4 BM25<br>• Cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`) |
| **Generation module** | `rag_tool.generate_answer` – OpenAI LLM with structured JSON output `{answer, sources, confidence}` |
| **Agentic workflow** | `workflow.py` (LangGraph)<br>• **Planner-executor:** the planner writes a plan and the executor follows it via conditional edges<br>• **Agents as Tools:** the Planner calls the Query Understanding Agent as a tool; the Escalation Agent calls the Sentiment Agent as a tool<br>• **Memory:** `MemorySaver` checkpointer keyed by `session_id`, so follow-up questions are rewritten using the conversation history |
| **RAG as a tool** | • `rag_search` / `rag_answer` are LangChain `@tool`s used by the RAG Retrieval Agent<br>• The Validator checks relevance and grounding before the final answer<br>• Tool chaining: RAG output → Validator → Responder → Escalation |
| **Feedback loop** | • Validator verdicts are logged<br>• On failure, the query is re-sent with a rewritten query and k + 2, up to 2 retries<br>• User 👍/👎 and corrections are stored in `data/feedback.jsonl`<br>• Corrections are re-ingested into the KB on the next index build |
| **Evaluation** | `evaluate.py` compares **Baseline RAG** (semantic top-k + one LLM call) with **Agentic RAG** (full graph) on:<br>• BLEU<br>• ROUGE-1, ROUGE-2, ROUGE-L<br>• Retrieval relevance (intent precision@k)<br>• Answer relevance (query–answer cosine similarity) |

**Agent roles (Use Case C)**

| Agent | Responsibility |
|---|---|
| Planner | Builds the step plan (retrieve → validate → respond → optional escalate) |
| Query Understanding Agent | Classifies intent and category; rewrites follow-ups into standalone queries |
| RAG Retrieval Agent | Calls the RAG tool to fetch FAQs and policies |
| Validator | Checks relevance and grounding; triggers a re-query on failure |
| Responder | Generates the grounded JSON answer |
| Sentiment Analysis Agent | Scores sentiment, urgency and satisfaction |
| Escalation Agent | Detects unresolved or angry cases and routes them to human support |

### Steps to run Section 1

1. **Create the environment and install dependencies.**
   ```bash
   python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env                                     # then add your OPENAI_API_KEY
   ```
2. **Build the vector store.** This downloads the dataset, cleans and chunks it, embeds it and saves the FAISS index.
   ```bash
   python -m src.ingestion.build_index
   ```
   📸 Screenshot: *vector store creation*
3. **Run the agent graph.** This prints the Mermaid flow diagram and a two-turn demo that shows memory working.
   ```bash
   python -m src.graph.workflow
   ```
   📸 Screenshots: *LangGraph flow diagram* and *agent trace output*
4. **Run the evaluation.** Use `--n 27` for a quick run.
   ```bash
   python -m src.evaluation.evaluate
   ```
   📸 Screenshot: *evaluation tables*. Results are saved to `data/eval_summary.csv`.
5. **Optional:** walk through every step in `notebooks/section1_experiments.ipynb`, which also covers the feedback demo.

---

## 4. Section 2 – FastAPI and Streamlit ✅ (built)

| PDF task | Implementation |
|---|---|
| **FastAPI backend** | `api/main.py` with Pydantic schemas in `api/schemas.py`<br>• `POST /query` – takes `{session_id?, query}` and returns `{session_id, answer, sources, retrieved_context, agents_invoked, confidence}`<br>• `GET /health` – status, LLM model and number of indexed vectors<br>• `POST /feedback` – takes 👍/👎 plus an optional correction and stores it in `data/feedback.jsonl` |
| **Streamlit frontend** | `ui/main.py` – a WhatsApp-style chat:<br>• Green user bubbles and white bot bubbles<br>• Retrieved context shown in bubbles with source ids<br>• Confidence and agent trace shown under each answer<br>• 👍/👎 buttons that call `/feedback`<br>• Calls the API with `requests.post()` |
| **Docker** | `api/Dockerfile` (CPU-only torch, models baked in), `ui/Dockerfile`, and `docker-compose.yml` to run both together |

Example request (`POST /query`):
```json
{"session_id": "abc123", "query": "How can I cancel my order 12345?"}
```
Example response (shortened):
```json
{"session_id": "abc123",
 "answer": "To cancel your order 12345, please follow these steps: 1. Log in to your account ...",
 "sources": ["bitext_faq#291", "bitext_faq#219", "bitext_faq#75", "bitext_faq#225"],
 "retrieved_context": ["Customer question: ... Support answer: ...", "..."],
 "agents_invoked": ["planner", "query_understanding", "rag_retrieval", "validator", "responder"],
 "confidence": 0.9}
```

### Steps to run Section 2

**Chat UI:** a WhatsApp-style chat with typing indicator, quick-reply chips, timestamps and ticks, an escalation pill, a 📎 Sources toggle, and 👍/👎 reactions (👎 lets you suggest a corrected answer).

**Run locally:** double-click `run_section2.bat`, or run these two commands in two separate terminals:
```bash
.venv\Scripts\activate
uvicorn api.main:app --port 8000           # Swagger UI: http://localhost:8000/docs
streamlit run ui/main.py                   # Chat UI:    http://localhost:8501
python -m api.smoke_test                   # optional: tests /health, /query and /feedback
```

**Run in Docker:** first install Docker Desktop, then:
```bash
docker compose up --build
```
- Chat UI: http://localhost:8501
- Swagger UI: http://localhost:8000/docs
- Section 1 must have been run first, because the API reads the FAISS index from `data/`.

📸 Screenshots: *Swagger UI* and *Streamlit chat UI*

## 5. Section 3 – CI/CD with GitHub Actions ✅ (built)

The pipeline lives in `.github/workflows/ci-cd.yml` and has three jobs:

| Job | Runs on | Steps |
|---|---|---|
| **test** | push + pull request | 1. Install dependencies<br>2. Lint with `ruff check .`<br>3. Run the `pytest` unit tests:<br>• `tests/test_retriever.py` – BM25, semantic, hybrid fusion, cleaning, splitting, chunking<br>• `tests/test_tools.py` – RAG tool, JSON generation, validator, feedback store<br>• `tests/test_api.py` – `/health`, `/query`, `/feedback`, input validation<br>The LLM and vector store are faked, so no API key is needed. |
| **build** | push to `main` | Build the API and UI Docker images and push them to **GitHub Container Registry** as `:<commit-sha>` and `:latest`, logging in with the `GITHUB_TOKEN` repository secret |
| **deploy** | after build | 1. Pull the new images.<br>2. Build the FAISS index inside the API image.<br>3. Run both containers on the Actions runner.<br>4. **Validate:** run `api/smoke_test.py` against `/health`, `/query` and `/feedback`, and check the Streamlit health endpoint.<br>5. On success, **promote** the images to `:stable`.<br>6. On failure, **roll back** to the previous `:stable` images and re-validate. |

**Optional secret:** to validate `/query` against the real LLM during deploy, add a repository secret named `OPENAI_API_KEY` (GitHub repo → Settings → Secrets and variables → Actions → New repository secret). Without it, the deploy step validates `/health` and `/feedback` only.

Run the tests locally with:
```bash
pip install pytest httpx ruff
ruff check .
pytest -v
```

📸 Screenshots: *GitHub Actions run (passing)* (Actions tab) and *deployed endpoint testing* (the "Validate deployment" step log)

## 6. Report checklist (PDF §8.1)

| # | Report section | Where the material comes from |
|---|---|---|
| 1 | Problem statement and use case | §1 above |
| 2 | Dataset, chunking and embeddings | §1 and `build_index.py` |
| 3 | Retrieval design (semantic, hybrid, re-ranking) | `retriever.py` |
| 4 | Agent design: roles, state, graph, Agents as Tools, memory, validation | `agents.py` and `workflow.py` |
| 5 | Feedback loop design | `feedback.py` and the retry loop |
| 6 | Evaluation tables | `eval_summary.csv` |
| 7 | API and UI design with request/response examples | Section 2 |
| 8 | Conclusion, future work and references | Written last |
