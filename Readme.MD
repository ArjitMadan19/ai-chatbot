# 🤖 AI PDF Chatbot (RAG)

An AI-powered chatbot that can answer questions from PDFs using Retrieval-Augmented Generation (RAG).

## 🚀 Features
- Chat with any PDF
- Semantic search using FAISS
- Local LLM (Ollama) — no API cost
- Built with LangChain

## 🧠 Tech Stack
- Python
- LangChain
- FAISS
- Ollama
- Sentence Transformers

## ⚙️ Setup

## Backend structure
```text
backend/
  app/
    api/       # FastAPI routes
    rag/       # RAG pipeline, FAISS, reranking, LLM calls
    services/  # config, database, Redis cache, upload/indexing helpers
    schemas/   # Pydantic request/response models
    scripts/   # ingestion and evaluation scripts
    tests/     # backend tests
    main.py    # FastAPI app factory and ASGI app
```

### 1. Clone repo
```bash
git clone https://github.com/your-username/ai-pdf-chatbot.git
cd ai-pdf-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the vector database
```bash
python -m backend.app.scripts.ingest
```

### 4. Configure Postgres
Create a local Postgres database:
```bash
createdb ai_chatbot
```

Create a `.env` file:
```bash
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/ai_chatbot
```

You can copy the full config template:
```bash
cp .env.example .env
```

`config.py` loads settings from `.env`, including the database URL, Redis URL, model names, vectorstore path, upload folder, chunk size, retrieval count, API memory turns, cache TTL, and LLM token limit.

The API is rate limited with SlowAPI. By default each client gets:
```bash
RATE_LIMIT=10/minute
```

Tables are created automatically when the FastAPI app starts.

### 5. Optional Redis cache
Repeated `/ask` calls are cached in Redis when Redis is available. The app still works if Redis is down.

Start Redis locally with Docker:
```bash
docker run --name ai-chatbot-redis -p 6379:6379 -d redis:7
```

Set cache config in `.env`:
```bash
CACHE_ENABLED=true
CACHE_TTL_SECONDS=300
REDIS_URL=redis://localhost:6379/0
```

### 6. Run the FastAPI server
```bash
uvicorn backend.app.main:api --reload
```

### 7. Ask a question
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the agreement say about confidentiality?", "doc_type_filter": "contract"}'
```

The response includes a `session_id`. Send it with later questions to save them in the same chat session and use recent messages as conversation memory:
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id": 1, "question": "Summarize that answer", "doc_type_filter": "contract"}'
```

View saved messages for a session:
```bash
curl http://127.0.0.1:8000/sessions/1
```

### 8. Upload a document
```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "doc_type=contract" \
  -F "file=@/path/to/document.pdf"
```

Uploads return immediately with `202 Accepted` and are indexed in a FastAPI background task. Check ingestion status:
```bash
curl http://127.0.0.1:8000/documents/1
```

Open the interactive API docs at:
```text
http://127.0.0.1:8000/docs
```
