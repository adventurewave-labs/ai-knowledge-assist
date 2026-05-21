# ARIA Onboarding Guide

Get a local development instance running in under 30 minutes.

## Prerequisites

- Python 3.11+
- `pip` or `uv`
- An OpenAI API key

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/marcuspat/ai-knowledge-assist.git
cd ai-knowledge-assist

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Start the API server (development)
uvicorn app.main:app --reload --port 8000
```

> **Production:** Do not use `--reload`. Run with:
> ```bash
> uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
> ```
> Or via Docker:
> ```bash
> docker build -t aria-rag .
> docker run -p 8000:8000 --env-file .env aria-rag
> ```

The API is now available at `http://localhost:8000`. Interactive docs at `/docs`.

## Ingest a Document

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "# My Document\n\nThis is knowledge I want to query.",
    "source": "my_doc.md"
  }'
```

## Query the Knowledge Base

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is in my document?"}'
```

## Run Tests

```bash
pytest tests/ -v
```

## Optional: Enable LangSmith Tracing

Set `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` in `.env`, then restart the server.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `GOOGLE_API_KEY` | `""` | Google API key for Gemini fallback |
| `LANGSMITH_API_KEY` | `""` | LangSmith tracing key |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store persistence path |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `LLM_MODEL` | `gpt-4o-mini` | Primary LLM model ID |
| `CHUNK_SIZE` | `512` | Max characters per document chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Default number of retrieved documents |
