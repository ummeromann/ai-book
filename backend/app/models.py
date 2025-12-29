"""Pydantic models for API requests and responses."""
from pydantic import BaseModel, EmailStr, Field
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


# Authentication Models
class UserSignupRequest(BaseModel):
    """Request model for user signup."""

    email: EmailStr
    password: str = Field(..., min_length=8)

    # Software background
    software_level: str = Field(..., pattern="^(Beginner|Intermediate|Advanced)$")
    languages_known: List[str] = []
    ai_experience: str = Field(..., pattern="^(None|Basic|Intermediate|Advanced)$")

    # Hardware background
    laptop_specs: Optional[str] = None
    gpu_available: Optional[str] = Field(None, pattern="^(RTX|No GPU)$")
    robotics_hardware: List[str] = []

    # Learning goals
    learning_goals: Optional[str] = None


class UserSigninRequest(BaseModel):
    """Request model for user signin."""

    email: EmailStr
    password: str


class UserProfileUpdate(BaseModel):
    """Request model for updating user profile."""

    # Software background
    software_level: Optional[str] = Field(None, pattern="^(Beginner|Intermediate|Advanced)$")
    languages_known: Optional[List[str]] = None
    ai_experience: Optional[str] = Field(None, pattern="^(None|Basic|Intermediate|Advanced)$")

    # Hardware background
    laptop_specs: Optional[str] = None
    gpu_available: Optional[str] = Field(None, pattern="^(RTX|No GPU)$")
    robotics_hardware: Optional[List[str]] = None

    # Learning goals
    learning_goals: Optional[str] = None


class UserProfileResponse(BaseModel):
    """Response model for user profile."""

    # Software background
    software_level: str
    languages_known: List[str]
    ai_experience: str

    # Hardware background
    laptop_specs: Optional[str]
    gpu_available: Optional[str]
    robotics_hardware: List[str]

    # Learning goals
    learning_goals: Optional[str]

    created_at: datetime
    updated_at: datetime


class UserResponse(BaseModel):
    """Response model for user information."""

    id: int
    email: str
    is_active: bool
    created_at: datetime
    profile: Optional[UserProfileResponse] = None


class AuthResponse(BaseModel):
    """Response model for authentication endpoints."""

    access_token: str
    token_type: str = "bearer"
    user: UserResponse
