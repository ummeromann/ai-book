"""Pydantic models for API requests and responses."""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    query: str
    session_id: Optional[str] = None
    max_results: Optional[int] = 5


class SelectedTextChatRequest(BaseModel):
    """Request model for selected-text chat endpoint."""

    query: str
    selected_text: str
    session_id: Optional[str] = None


class ChatMessage(BaseModel):
    """Individual chat message."""

    role: str  # 'user' or 'assistant'
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""

    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    metadata: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    """Request model for content ingestion."""

    content_path: Optional[str] = None  # Path to book content directory
    force_reingest: bool = False


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""

    status: str
    files_processed: int
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    qdrant_connected: bool
    database_connected: bool
