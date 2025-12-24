"""RAG service for retrieval and generation using Qdrant and OpenAI."""
import os
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from app.config import settings
from app.database import BookContent


class RAGService:
    """Service for RAG operations including ingestion, retrieval, and generation."""

    def __init__(self):
        """Initialize RAG service with Qdrant and OpenAI clients."""
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.collection_name = settings.qdrant_collection_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure Qdrant collection exists."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name not in collection_names:
            # Create collection with embedding dimension (1536 for text-embedding-3-small)
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text using OpenAI."""
        response = self.openai_client.embeddings.create(
            model=settings.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def ingest_book_content(self, content_path: str, db: Session, force_reingest: bool = False) -> Dict[str, Any]:
        """
        Ingest book content from markdown files.

        Args:
            content_path: Path to directory containing book markdown files
            db: Database session
            force_reingest: If True, reingest even if content hasn't changed

        Returns:
            Dictionary with ingestion statistics
        """
        files_processed = 0
        chunks_created = 0
        points = []
        point_id = 0

        # Find all markdown files
        content_dir = Path(content_path)
        md_files = list(content_dir.rglob("*.md"))

        for md_file in md_files:
            # Read file content
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Compute content hash
            content_hash = self._compute_content_hash(content)

            # Check if file already ingested
            existing_entry = db.query(BookContent).filter_by(file_path=str(md_file)).first()
            if existing_entry and existing_entry.content_hash == content_hash and not force_reingest:
                continue  # Skip if content hasn't changed

            # Extract metadata from file path
            relative_path = md_file.relative_to(content_dir)
            parts = str(relative_path).split(os.sep)

            metadata = {
                "file_path": str(md_file),
                "relative_path": str(relative_path),
                "filename": md_file.name,
            }

            # Try to extract module/chapter info
            if len(parts) >= 2:
                metadata["module"] = parts[0]
                if len(parts) >= 3:
                    metadata["chapter"] = parts[1]

            # Split content into chunks
            chunks = self.text_splitter.split_text(content)

            # Create embeddings and points for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                embedding = self._get_embedding(chunk)

                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_idx,
                    "chunk_text": chunk,
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=chunk_metadata,
                    )
                )
                point_id += 1
                chunks_created += 1

            # Update or create database entry
            if existing_entry:
                existing_entry.content_hash = content_hash
                existing_entry.metadata = metadata
            else:
                new_entry = BookContent(
                    file_path=str(md_file),
                    title=md_file.stem,
                    module=metadata.get("module"),
                    chapter=metadata.get("chapter"),
                    content_hash=content_hash,
                    metadata=metadata,
                )
                db.add(new_entry)

            files_processed += 1

        # Upload points to Qdrant in batches
        if points:
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )

        db.commit()

        return {
            "files_processed": files_processed,
            "chunks_created": chunks_created,
        }

    def retrieve_relevant_chunks(
        self,
        query: str,
        max_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from Qdrant.

        Args:
            query: User query
            max_results: Maximum number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Build filter if provided
        query_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)

        # Search Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=max_results,
            query_filter=query_filter,
        )

        # Format results
        chunks = []
        for result in search_results:
            chunks.append({
                "text": result.payload.get("chunk_text", ""),
                "score": result.score,
                "metadata": {
                    "file_path": result.payload.get("file_path", ""),
                    "module": result.payload.get("module", ""),
                    "chapter": result.payload.get("chapter", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                },
            })

        return chunks

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        selected_text: Optional[str] = None
    ) -> str:
        """
        Generate answer using OpenAI based on retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            selected_text: Optional selected text for focused answering

        Returns:
            Generated answer
        """
        # Build context from chunks
        if selected_text:
            context = f"Selected Text:\n{selected_text}\n\n"
            system_message = (
                "You are a helpful AI assistant for the Physical AI & Humanoid Robotics book. "
                "Answer questions STRICTLY based on the selected text provided. "
                "If the answer cannot be found in the selected text, say so explicitly. "
                "Always cite the relevant parts of the selected text in your answer."
            )
        else:
            context = "Context from the book:\n\n"
            for i, chunk in enumerate(context_chunks, 1):
                module = chunk["metadata"].get("module", "Unknown")
                chapter = chunk["metadata"].get("chapter", "Unknown")
                context += f"[Source {i} - {module}/{chapter}]:\n{chunk['text']}\n\n"

            system_message = (
                "You are a helpful AI assistant for the Physical AI & Humanoid Robotics book. "
                "Answer questions based ONLY on the context provided from the book. "
                "If the answer cannot be found in the provided context, say so explicitly. "
                "Always cite the relevant source numbers (e.g., [Source 1]) when answering. "
                "Do not make up information or use knowledge outside the provided context."
            )

        # Generate answer using OpenAI
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
        ]

        response = self.openai_client.chat.completions.create(
            model=settings.chat_model,
            messages=messages,
            temperature=0.3,  # Lower temperature for more factual responses
        )

        return response.choices[0].message.content

    def answer_with_selected_text(self, query: str, selected_text: str) -> Dict[str, Any]:
        """
        Answer question based strictly on selected text.

        Args:
            query: User query
            selected_text: Selected text from the book

        Returns:
            Answer and metadata
        """
        # Generate answer directly from selected text
        answer = self.generate_answer(query, [], selected_text=selected_text)

        return {
            "answer": answer,
            "sources": [{
                "type": "selected_text",
                "text": selected_text[:200] + "..." if len(selected_text) > 200 else selected_text,
            }],
        }

    def answer_with_retrieval(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Answer question using RAG retrieval from entire book.

        Args:
            query: User query
            max_results: Maximum number of chunks to retrieve

        Returns:
            Answer and sources
        """
        # Retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(query, max_results=max_results)

        if not chunks:
            return {
                "answer": "I couldn't find relevant information in the book to answer your question.",
                "sources": [],
            }

        # Generate answer
        answer = self.generate_answer(query, chunks)

        # Format sources
        sources = [
            {
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "score": chunk["score"],
                "module": chunk["metadata"].get("module", ""),
                "chapter": chunk["metadata"].get("chapter", ""),
                "file_path": chunk["metadata"].get("file_path", ""),
            }
            for chunk in chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
        }
