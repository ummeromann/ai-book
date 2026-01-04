"""Test RAG service directly"""
import sys
sys.path.insert(0, '.')

from app.rag_service import RAGService
from app.config import settings

print(f"OpenRouter API Key present: {bool(settings.openrouter_api_key)}")
print(f"Use Fallback: {settings.use_fallback}")

# Initialize RAG service
rag = RAGService()
print(f"OpenRouter client initialized: {rag.openrouter_client is not None}")

# Try a simple query
try:
    result = rag.answer_with_retrieval("Hello, test message", max_results=3)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
