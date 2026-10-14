# Streamlit chat UI image (build from the project root: docker build -f ui/Dockerfile -t rag-ui .)
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install streamlit requests
COPY ui/main.py ui/main.py
EXPOSE 8501
CMD ["streamlit", "run", "ui/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
