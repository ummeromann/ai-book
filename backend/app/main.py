"""Main FastAPI application."""
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.database import init_db, get_db, ChatSession, ChatMessage as DBChatMessage, User, UserProfile
from app.models import (
    ChatRequest,
    SelectedTextChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    UserSignupRequest,
    UserSigninRequest,
    UserProfileUpdate,
    UserProfileResponse,
    UserResponse,
    AuthResponse,
)
from app.rag_service import RAGService
from app.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    get_current_user_optional,
)

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

# Global RAG service instance
rag_service = None


# Initialize database and RAG service on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables and RAG service on startup."""
    global rag_service
    init_db()
    rag_service = RAGService()
    print(f"[Startup] RAG Service initialized with demo_mode={rag_service.demo_mode}")


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


@app.post("/auth/signup", response_model=AuthResponse, tags=["Authentication"])
async def signup(
    request: UserSignupRequest,
    db: Session = Depends(get_db),
):
    """
    User signup endpoint with profile creation.

    Creates a new user account and associated profile with learning preferences.
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create user
    hashed_password = get_password_hash(request.password)
    new_user = User(
        email=request.email,
        hashed_password=hashed_password,
    )
    db.add(new_user)
    db.flush()  # Flush to get the user ID

    # Create user profile
    user_profile = UserProfile(
        user_id=new_user.id,
        software_level=request.software_level,
        languages_known=request.languages_known,
        ai_experience=request.ai_experience,
        laptop_specs=request.laptop_specs,
        gpu_available=request.gpu_available,
        robotics_hardware=request.robotics_hardware,
        learning_goals=request.learning_goals,
    )
    db.add(user_profile)
    db.commit()
    db.refresh(new_user)

    # Create access token
    access_token = create_access_token(
        data={"sub": new_user.id},
        expires_delta=timedelta(days=settings.access_token_expire_days)
    )

    # Build response
    profile_response = UserProfileResponse(
        software_level=user_profile.software_level,
        languages_known=user_profile.languages_known or [],
        ai_experience=user_profile.ai_experience,
        laptop_specs=user_profile.laptop_specs,
        gpu_available=user_profile.gpu_available,
        robotics_hardware=user_profile.robotics_hardware or [],
        learning_goals=user_profile.learning_goals,
        created_at=user_profile.created_at,
        updated_at=user_profile.updated_at,
    )

    user_response = UserResponse(
        id=new_user.id,
        email=new_user.email,
        is_active=new_user.is_active,
        created_at=new_user.created_at,
        profile=profile_response,
    )

    return AuthResponse(
        access_token=access_token,
        user=user_response,
    )


@app.post("/auth/signin", response_model=AuthResponse, tags=["Authentication"])
async def signin(
    request: UserSigninRequest,
    db: Session = Depends(get_db),
):
    """
    User signin endpoint.

    Authenticates user and returns access token with user profile.
    """
    # Find user
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    # Create access token
    access_token = create_access_token(
        data={"sub": user.id},
        expires_delta=timedelta(days=settings.access_token_expire_days)
    )

    # Build response with profile
    profile_response = None
    if user.profile:
        profile_response = UserProfileResponse(
            software_level=user.profile.software_level,
            languages_known=user.profile.languages_known or [],
            ai_experience=user.profile.ai_experience,
            laptop_specs=user.profile.laptop_specs,
            gpu_available=user.profile.gpu_available,
            robotics_hardware=user.profile.robotics_hardware or [],
            learning_goals=user.profile.learning_goals,
            created_at=user.profile.created_at,
            updated_at=user.profile.updated_at,
        )

    user_response = UserResponse(
        id=user.id,
        email=user.email,
        is_active=user.is_active,
        created_at=user.created_at,
        profile=profile_response,
    )

    return AuthResponse(
        access_token=access_token,
        user=user_response,
    )


@app.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """Get current authenticated user information."""
    profile_response = None
    if current_user.profile:
        profile_response = UserProfileResponse(
            software_level=current_user.profile.software_level,
            languages_known=current_user.profile.languages_known or [],
            ai_experience=current_user.profile.ai_experience,
            laptop_specs=current_user.profile.laptop_specs,
            gpu_available=current_user.profile.gpu_available,
            robotics_hardware=current_user.profile.robotics_hardware or [],
            learning_goals=current_user.profile.learning_goals,
            created_at=current_user.profile.created_at,
            updated_at=current_user.profile.updated_at,
        )

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        profile=profile_response,
    )


@app.put("/auth/profile", response_model=UserProfileResponse, tags=["Authentication"])
async def update_user_profile(
    request: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update user profile information."""
    if not current_user.profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found"
        )

    # Update profile fields
    if request.software_level is not None:
        current_user.profile.software_level = request.software_level
    if request.languages_known is not None:
        current_user.profile.languages_known = request.languages_known
    if request.ai_experience is not None:
        current_user.profile.ai_experience = request.ai_experience
    if request.laptop_specs is not None:
        current_user.profile.laptop_specs = request.laptop_specs
    if request.gpu_available is not None:
        current_user.profile.gpu_available = request.gpu_available
    if request.robotics_hardware is not None:
        current_user.profile.robotics_hardware = request.robotics_hardware
    if request.learning_goals is not None:
        current_user.profile.learning_goals = request.learning_goals

    current_user.profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(current_user.profile)

    return UserProfileResponse(
        software_level=current_user.profile.software_level,
        languages_known=current_user.profile.languages_known or [],
        ai_experience=current_user.profile.ai_experience,
        laptop_specs=current_user.profile.laptop_specs,
        gpu_available=current_user.profile.gpu_available,
        robotics_hardware=current_user.profile.robotics_hardware or [],
        learning_goals=current_user.profile.learning_goals,
        created_at=current_user.profile.created_at,
        updated_at=current_user.profile.updated_at,
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
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Chat endpoint for answering questions about the entire book.

    This endpoint:
    1. Retrieves relevant chunks from Qdrant based on the query
    2. Generates an answer using OpenAI with the retrieved context
    3. Personalizes response based on user profile (if authenticated)
    4. Stores the conversation in Postgres
    5. Returns the answer with source citations
    """
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())

        # Extract user context for personalization
        user_context = None
        if current_user and current_user.profile:
            user_context = {
                "software_level": current_user.profile.software_level,
                "languages_known": current_user.profile.languages_known or [],
                "ai_experience": current_user.profile.ai_experience,
                "gpu_available": current_user.profile.gpu_available,
                "learning_goals": current_user.profile.learning_goals,
            }

        # In demo mode, skip database operations
        if not settings.demo_mode:
            # Check if session exists
            session = db.query(ChatSession).filter_by(session_id=session_id).first()
            if not session:
                session = ChatSession(
                    session_id=session_id,
                    user_id=current_user.id if current_user else None
                )
                db.add(session)
                db.commit()

            # Store user message
            user_message = DBChatMessage(
                session_id=session_id,
                role="user",
                content=request.query,
            )
            db.add(user_message)

        # Get answer using RAG with personalization
        result = rag_service.answer_with_retrieval(
            query=request.query,
            max_results=request.max_results or settings.max_retrieval_results,
            user_context=user_context,
        )

        # In demo mode, skip database storage
        if not settings.demo_mode:
            # Store assistant message
            assistant_message = DBChatMessage(
                session_id=session_id,
                role="assistant",
                content=result["answer"],
                message_metadata={"sources": result["sources"]},
            )
            db.add(assistant_message)

            # Update session timestamp
            session.updated_at = datetime.utcnow()
            db.commit()

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=session_id,
            metadata={"query": request.query, "demo_mode": settings.demo_mode},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/chat/selected", response_model=ChatResponse, tags=["Chat"])
async def chat_selected_text(
    request: SelectedTextChatRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Chat endpoint for answering questions based ONLY on user-selected text.

    This endpoint:
    1. Takes the user's selected text and query
    2. Generates an answer strictly from the selected text (no retrieval)
    3. Personalizes response based on user profile (if authenticated)
    4. Stores the conversation in Postgres
    5. Returns the answer with the selected text as the source
    """
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())

        # Extract user context for personalization
        user_context = None
        if current_user and current_user.profile:
            user_context = {
                "software_level": current_user.profile.software_level,
                "languages_known": current_user.profile.languages_known or [],
                "ai_experience": current_user.profile.ai_experience,
                "gpu_available": current_user.profile.gpu_available,
                "learning_goals": current_user.profile.learning_goals,
            }

        # In demo mode, skip database operations
        if not settings.demo_mode:
            # Check if session exists
            session = db.query(ChatSession).filter_by(session_id=session_id).first()
            if not session:
                session = ChatSession(
                    session_id=session_id,
                    user_id=current_user.id if current_user else None
                )
                db.add(session)
                db.commit()

            # Store user message with selected text metadata
            user_message = DBChatMessage(
                session_id=session_id,
                role="user",
                content=request.query,
                message_metadata={"selected_text": request.selected_text},
            )
            db.add(user_message)

        # Get answer based on selected text only with personalization
        result = rag_service.answer_with_selected_text(
            query=request.query,
            selected_text=request.selected_text,
            user_context=user_context,
        )

        # In demo mode, skip database storage
        if not settings.demo_mode:
            # Store assistant message
            assistant_message = DBChatMessage(
                session_id=session_id,
                role="assistant",
                content=result["answer"],
                message_metadata={"sources": result["sources"], "mode": "selected_text"},
            )
            db.add(assistant_message)

            # Update session timestamp
            session.updated_at = datetime.utcnow()
            db.commit()

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=session_id,
            metadata={"query": request.query, "mode": "selected_text", "demo_mode": settings.demo_mode},
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
                "metadata": msg.message_metadata,
                "created_at": msg.created_at.isoformat(),
            }
            for msg in messages
        ],
    }
