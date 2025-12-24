# Backend - RAG Chatbot API

FastAPI-based RAG (Retrieval-Augmented Generation) service for the Physical AI Book.

## Features

- **Full-Book RAG**: Answer questions using retrieval from the entire book
- **Selected-Text RAG**: Answer questions based strictly on user-selected text
- **Vector Search**: Qdrant Cloud for efficient semantic search
- **Conversation History**: Neon Serverless Postgres for session and message storage
- **OpenAI Integration**: GPT-4 for answer generation, text-embedding-3-small for embeddings

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Frontend   │────▶│  FastAPI     │────▶│   OpenAI    │
│ (Docusaurus)│     │   Backend    │     │  (GPT-4)    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ├──────▶ Qdrant Cloud (Vectors)
                           │
                           └──────▶ Neon Postgres (Metadata)
```

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `QDRANT_URL`: Qdrant Cloud cluster URL
- `QDRANT_API_KEY`: Qdrant API key
- `DATABASE_URL`: Neon Postgres connection string

### 3. Run the Server

```bash
python run.py
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### 4. Ingest Book Content

Before using the chatbot, ingest the book content:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"content_path": "../frontend/docs", "force_reingest": false}'
```

## API Endpoints

### Health Check
- `GET /health` - Check service health

### Ingestion
- `POST /ingest` - Ingest book content into RAG system

### Chat
- `POST /chat` - Ask questions about the entire book
- `POST /chat/selected` - Ask questions about selected text only
- `GET /sessions/{session_id}/history` - Get conversation history

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration management
│   ├── database.py       # Database models and connection
│   ├── models.py         # Pydantic models
│   └── rag_service.py    # RAG logic (retrieval + generation)
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
├── run.py               # Server startup script
└── README.md            # This file
```

## How It Works

### Full-Book RAG Mode

1. User sends a query to `/chat`
2. Query is embedded using OpenAI embeddings
3. Top-K relevant chunks retrieved from Qdrant
4. Chunks + query sent to GPT-4 for answer generation
5. Answer returned with source citations
6. Conversation saved to Postgres

### Selected-Text Mode

1. User selects text in the book and asks a question
2. Request sent to `/chat/selected` with query + selected text
3. GPT-4 generates answer STRICTLY from selected text (no retrieval)
4. Answer returned with selected text as source
5. Conversation saved to Postgres

## Development

### Running Tests

```bash
pytest
```

### Database Migrations

The application uses SQLAlchemy and automatically creates tables on startup.

### Adding New Features

1. Add new endpoints in `app/main.py`
2. Add business logic in `app/rag_service.py`
3. Add models in `app/models.py` and `app/database.py`
