"""Database models and connection management."""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.config import settings

Base = declarative_base()


class ChatSession(Base):
    """Chat session model for storing conversation history."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True)
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


# Database engine and session
# Configure connection arguments for Neon with proper SSL handling for psycopg3
connect_args = {}
if "neon.tech" in settings.database_url:
    connect_args = {
        "connect_timeout": 10,
        "options": "-c timezone=utc",
        # Disable SSL verification for pooled connections (Neon specific)
        "sslmode": "require",
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,  # Verify connections before using them
    pool_recycle=300,  # Recycle connections after 5 minutes
    pool_size=5,  # Reduce pool size for better connection management
    max_overflow=10,
)
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
