# ECM AI Platform Backend

A prototype-first FastAPI backend for an Enterprise Content Management (ECM) AI platform.

## What is included

- FastAPI application with OpenAPI/Swagger support
- Upload pipeline for PDF and image documents
- OCR extraction and fallback behavior
- Chunking, embeddings, and ChromaDB indexing
- RAG pipeline for retrieval-augmented generation
- Groq integration for answer generation
- Semantic search with metadata filtering and top-k control
- Chat API for question answering
- Lightweight workflow engine with pending / approved / rejected states
- Logging and defensive error handling

## Demo flow

1. Upload a PDF or image document
2. The backend extracts text, chunks it, generates embeddings, and stores vectors in ChromaDB
3. Ask questions through `/chat`
4. Search semantically through `/search`
5. Track approval workflows through `/workflow`

## Project structure

```text
backend/
├── app/
│   ├── main.py
│   ├── ai/
│   │   ├── extractor.py
│   │   ├── llm_client.py
│   │   └── ocr.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── rag/
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── pipeline.py
│   │   ├── retriever.py
│   │   ├── vector_store.py
│   │   └── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── health.py
│   │   ├── ocr.py
│   │   ├── rag.py
│   │   ├── search.py
│   │   ├── upload.py
│   │   └── workflow.py
│   ├── services/
│   │   ├── chat_service.py
│   │   ├── document_service.py
│   │   ├── indexing_service.py
│   │   ├── ocr_service.py
│   │   ├── search_service.py
│   │   └── workflow_service.py
│   ├── utils/
│   │   ├── file_handler.py
│   │   └── file_utils.py
│   └── uploads/
├── requirements.txt
├── .env
└── README.md
```

## Startup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open the API docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment variables

The `.env` file contains prototype defaults for:

- `APP_ENV`
- `LOG_LEVEL`
- `BACKEND_HOST`
- `BACKEND_PORT`
- `ALLOWED_ORIGINS`
- `UPLOAD_DIR`
- `MODEL_NAME`
- `MODEL_API_KEY`
- `GROQ_MODEL`
- `GROQ_API_KEY`
- `MODEL_BASE_URL`
- `MODEL_TIMEOUT`
- `MODEL_TEMPERATURE`

## Prototype flow

```text
Upload PDF or image
  -> OCR / extraction
  -> chunking
  -> embeddings
  -> ChromaDB indexing
  -> semantic search
  -> chat + citations
  -> workflow actions
```

## Demo endpoints

- `POST /upload` – upload and index a document
- `POST /chat` – ask a question against indexed content
- `POST /chat/stream` – stream chat output
- `GET /search` – run semantic search with metadata filtering
- `POST /workflow` – create a workflow
- `POST /workflow/{document_id}/approve` – approve a workflow
- `POST /workflow/{document_id}/reject` – reject a workflow
- `GET /workflow/{document_id}` – get workflow status
- `POST /ocr/extract` – inspect OCR extraction output

## Prototype notes

- The backend is intentionally lightweight and optimized for a demo environment.
- Authentication, advanced security, and Kubernetes deployment are not included.
- ChromaDB is used as the local vector store for fast prototype iteration.
- GROQ usage is controlled through `.env` and the existing RAG pipeline.
