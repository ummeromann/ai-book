"""Main FastAPI application."""
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.database import init_db, get_db, ChatSession, ChatMessage as DBChatMessage
from app.models import (
    ChatRequest,
    SelectedTextChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
)
from app.rag_service import RAGService

# Initialize FastAPI app
app = FastAPI(
    title="Physical AI Book RAG API",
    description="RAG-powered chatbot API for the Physical AI & Humanoid Robotics book",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup."""
    init_db()


# Initialize RAG service (singleton)
rag_service = RAGService()


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "Physical AI Book RAG API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    # Check Qdrant connection
    qdrant_connected = False
    try:
        rag_service.qdrant_client.get_collections()
        qdrant_connected = True
    except Exception:
        pass

    # Check database connection
    database_connected = False
    try:
        db.execute("SELECT 1")
        database_connected = True
    except Exception:
        pass

    status = "healthy" if (qdrant_connected and database_connected) else "degraded"

    return HealthResponse(
        status=status,
        qdrant_connected=qdrant_connected,
        database_connected=database_connected,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_content(
    request: IngestRequest,
    db: Session = Depends(get_db),
):
    """
    Ingest book content into the RAG system.

    This endpoint:
    1. Scans the book content directory for markdown files
    2. Chunks the content appropriately
    3. Generates embeddings using OpenAI
    4. Stores embeddings in Qdrant
    5. Stores metadata in Postgres
    """
    try:
        # Default to frontend/docs directory if no path provided
        content_path = request.content_path or "../frontend/docs"

        # Ingest content
        result = rag_service.ingest_book_content(
            content_path=content_path,
            db=db,
            force_reingest=request.force_reingest,
        )

        return IngestResponse(
            status="success",
            files_processed=result["files_processed"],
            chunks_created=result["chunks_created"],
            message=f"Successfully ingested {result['files_processed']} files and created {result['chunks_created']} chunks.",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Chat endpoint for answering questions about the entire book.

    This endpoint:
    1. Retrieves relevant chunks from Qdrant based on the query
    2. Generates an answer using OpenAI with the retrieved context
    3. Stores the conversation in Postgres
    4. Returns the answer with source citations
    """
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())

        # Check if session exists
        session = db.query(ChatSession).filter_by(session_id=session_id).first()
        if not session:
            session = ChatSession(session_id=session_id)
            db.add(session)
            db.commit()

        # Store user message
        user_message = DBChatMessage(
            session_id=session_id,
            role="user",
            content=request.query,
        )
        db.add(user_message)

        # Get answer using RAG
        result = rag_service.answer_with_retrieval(
            query=request.query,
            max_results=request.max_results or settings.max_retrieval_results,
        )

        # Store assistant message
        assistant_message = DBChatMessage(
            session_id=session_id,
            role="assistant",
            content=result["answer"],
            metadata={"sources": result["sources"]},
        )
        db.add(assistant_message)

        # Update session timestamp
        session.updated_at = datetime.utcnow()
        db.commit()

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=session_id,
            metadata={"query": request.query},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/chat/selected", response_model=ChatResponse, tags=["Chat"])
async def chat_selected_text(
    request: SelectedTextChatRequest,
    db: Session = Depends(get_db),
):
    """
    Chat endpoint for answering questions based ONLY on user-selected text.

    This endpoint:
    1. Takes the user's selected text and query
    2. Generates an answer strictly from the selected text (no retrieval)
    3. Stores the conversation in Postgres
    4. Returns the answer with the selected text as the source
    """
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())

        # Check if session exists
        session = db.query(ChatSession).filter_by(session_id=session_id).first()
        if not session:
            session = ChatSession(session_id=session_id)
            db.add(session)
            db.commit()

        # Store user message with selected text metadata
        user_message = DBChatMessage(
            session_id=session_id,
            role="user",
            content=request.query,
            metadata={"selected_text": request.selected_text},
        )
        db.add(user_message)

        # Get answer based on selected text only
        result = rag_service.answer_with_selected_text(
            query=request.query,
            selected_text=request.selected_text,
        )

        # Store assistant message
        assistant_message = DBChatMessage(
            session_id=session_id,
            role="assistant",
            content=result["answer"],
            metadata={"sources": result["sources"], "mode": "selected_text"},
        )
        db.add(assistant_message)

        # Update session timestamp
        session.updated_at = datetime.utcnow()
        db.commit()

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=session_id,
            metadata={"query": request.query, "mode": "selected_text"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Selected text chat failed: {str(e)}")


@app.get("/sessions/{session_id}/history", tags=["Chat"])
async def get_session_history(
    session_id: str,
    db: Session = Depends(get_db),
):
    """Get chat history for a specific session."""
    messages = (
        db.query(DBChatMessage)
        .filter_by(session_id=session_id)
        .order_by(DBChatMessage.created_at)
        .all()
    )

    return {
        "session_id": session_id,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "metadata": msg.metadata,
                "created_at": msg.created_at.isoformat(),
            }
            for msg in messages
        ],
    }
