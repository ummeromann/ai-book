"""Application configuration using Pydantic settings."""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4-turbo-preview"

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "physical_ai_book"

    # Neon Postgres
    database_url: str

    # RAG Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_results: int = 5

    # CORS
    allowed_origins: str = "http://localhost:3000"

    @property
    def origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
