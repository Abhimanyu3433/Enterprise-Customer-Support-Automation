# FastAPI backend image (build from the project root: docker build -f api/Dockerfile -t rag-api .)
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf_cache

COPY requirements.txt .
# CPU-only torch keeps the image small (no CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

# Bake the embedding + re-ranker models into the image so start-up needs no download
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY src/ src/
COPY api/ api/
# data/ (FAISS index, chunks, feedback) is mounted as a volume, see docker-compose.yml
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
