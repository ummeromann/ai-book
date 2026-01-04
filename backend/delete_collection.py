"""Delete the old Qdrant collection to create fresh one with correct dimensions."""
from app.config import settings
from qdrant_client import QdrantClient

print(f"Connecting to Qdrant at {settings.qdrant_url}...")
client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)

collection_name = settings.qdrant_collection_name

try:
    client.delete_collection(collection_name)
    print(f"✓ Deleted collection '{collection_name}'")
except Exception as e:
    print(f"Collection '{collection_name}' doesn't exist or couldn't be deleted: {e}")

print("Done!")
