"""Database models and connection management."""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from app.config import settings

Base = declarative_base()


class ChatSession(Base):
    """Chat session model for storing conversation history."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Optional: link to authenticated user
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatMessage(Base):
    """Chat message model for storing individual messages."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    role = Column(String(50))  # 'user' or 'assistant'
    content = Column(Text)
    message_metadata = Column(JSON, nullable=True)  # Store context chunks, selected text, etc.
    created_at = Column(DateTime, default=datetime.utcnow)


class BookContent(Base):
    """Book content metadata model."""

    __tablename__ = "book_content"

    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String(500), unique=True)
    title = Column(String(500))
    module = Column(String(100), nullable=True)
    chapter = Column(String(100), nullable=True)
    content_hash = Column(String(64))  # SHA-256 hash for change detection
    ingested_at = Column(DateTime, default=datetime.utcnow)
    content_metadata = Column(JSON, nullable=True)


class User(Base):
    """User model for authentication."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to profile
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")


class UserProfile(Base):
    """User profile model for personalization."""

    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    # Software background
    software_level = Column(String(50))  # Beginner / Intermediate / Advanced
    languages_known = Column(JSON, nullable=True)  # ["Python", "JavaScript", "ROS"]
    ai_experience = Column(String(50))  # None / Basic / Intermediate / Advanced

    # Hardware background
    laptop_specs = Column(Text, nullable=True)
    gpu_available = Column(String(50), nullable=True)  # RTX / No GPU
    robotics_hardware = Column(JSON, nullable=True)  # ["Jetson", "Arduino", etc.]

    # Learning goals
    learning_goals = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to user
    user = relationship("User", back_populates="profile")


# Database engine and session
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
