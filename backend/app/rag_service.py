"""RAG service for retrieval and generation using Qdrant and OpenAI."""
import os
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from app.config import settings
from app.database import BookContent


class RAGService:
    """Service for RAG operations including ingestion, retrieval, and generation."""

    def __init__(self):
        """Initialize RAG service with Qdrant and OpenAI clients."""
        import logging
        logger = logging.getLogger("uvicorn")

        self.demo_mode = settings.demo_mode
        logger.info(f"RAGService initializing with demo_mode={self.demo_mode}")

        if self.demo_mode:
            # Demo mode - no external services needed
            logger.info("Demo mode enabled - skipping OpenAI and Qdrant initialization")
            self.openai_client = None
            self.qdrant_client = None
            self.collection_name = None
            self.text_splitter = None
        else:
            logger.info("Production mode - initializing OpenAI and Qdrant clients")
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
                existing_entry.content_metadata = metadata
            else:
                new_entry = BookContent(
                    file_path=str(md_file),
                    title=md_file.stem,
                    module=metadata.get("module"),
                    chapter=metadata.get("chapter"),
                    content_hash=content_hash,
                    content_metadata=metadata,
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
        selected_text: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate answer using OpenAI based on retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            selected_text: Optional selected text for focused answering
            user_context: Optional user profile context for personalization

        Returns:
            Generated answer
        """
        # Build personalization instructions
        personalization = ""
        if user_context:
            personalization = "\n\nPersonalization Context:"
            personalization += f"\n- User's software level: {user_context.get('software_level', 'Unknown')}"
            personalization += f"\n- Programming experience: {', '.join(user_context.get('languages_known', [])) or 'Not specified'}"
            personalization += f"\n- AI/ML experience: {user_context.get('ai_experience', 'Unknown')}"
            if user_context.get('gpu_available'):
                personalization += f"\n- GPU available: {user_context['gpu_available']}"
            if user_context.get('learning_goals'):
                personalization += f"\n- Learning goals: {user_context['learning_goals']}"

            personalization += "\n\nAdjust your explanation difficulty, code examples, and recommendations based on the user's background."

        # Build context from chunks
        if selected_text:
            context = f"Selected Text:\n{selected_text}\n\n"
            system_message = (
                "You are a helpful AI assistant for the Physical AI & Humanoid Robotics book. "
                "Answer questions STRICTLY based on the selected text provided. "
                "If the answer cannot be found in the selected text, say so explicitly. "
                "Always cite the relevant parts of the selected text in your answer."
                + personalization
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
                + personalization
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

    def answer_with_selected_text(
        self,
        query: str,
        selected_text: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer question based strictly on selected text.

        Args:
            query: User query
            selected_text: Selected text from the book
            user_context: Optional user profile for personalization

        Returns:
            Answer and metadata
        """
        # Demo mode - return mock response
        if self.demo_mode:
            answer = f"""Based on the selected text, here's a demo response to your question: "{query}"

Selected text preview: {selected_text[:100]}{'...' if len(selected_text) > 100 else ''}

[This is a DEMO response - OpenAI integration is not active. In full mode, this would provide an AI-generated answer based strictly on your selected text.]"""
            return {
                "answer": answer,
                "sources": [{
                    "type": "selected_text",
                    "text": selected_text[:200] + "..." if len(selected_text) > 200 else selected_text,
                }],
            }

        # Generate answer directly from selected text with personalization
        answer = self.generate_answer(
            query,
            [],
            selected_text=selected_text,
            user_context=user_context
        )

        return {
            "answer": answer,
            "sources": [{
                "type": "selected_text",
                "text": selected_text[:200] + "..." if len(selected_text) > 200 else selected_text,
            }],
        }

    def answer_with_retrieval(
        self,
        query: str,
        max_results: int = 5,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer question using RAG retrieval from entire book.

        Args:
            query: User query
            max_results: Maximum number of chunks to retrieve
            user_context: Optional user profile for personalization

        Returns:
            Answer and sources
        """
        # Demo mode - return mock responses
        if self.demo_mode:
            return self._demo_answer(query)

        # Retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(query, max_results=max_results)

        if not chunks:
            return {
                "answer": "I couldn't find relevant information in the book to answer your question.",
                "sources": [],
            }

        # Generate answer with personalization
        answer = self.generate_answer(query, chunks, user_context=user_context)

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

    def _demo_answer(self, query: str) -> Dict[str, Any]:
        """
        Generate demo responses without calling OpenAI.

        Args:
            query: User query

        Returns:
            Demo answer and sources
        """
        # Contextual demo responses based on query keywords
        query_lower = query.lower()

        if any(word in query_lower for word in ['robot', 'humanoid', 'physical ai']):
            answer = """Physical AI and humanoid robotics represent the convergence of artificial intelligence with physical embodiment.

Key aspects include:
- Embodied intelligence that can interact with the real world
- Humanoid robots designed to navigate human environments
- Integration of perception, reasoning, and action
- Applications in manufacturing, healthcare, and service industries

[This is a DEMO response - OpenAI integration is not active]"""
            sources = [
                {
                    "text": "Physical AI combines artificial intelligence with robotics to create systems that can perceive and interact with the physical world...",
                    "score": 0.95,
                    "module": "module-0-foundations",
                    "chapter": "ch01-intro-physical-ai",
                },
                {
                    "text": "Humanoid robots are designed to mimic human form and function, making them well-suited for human environments...",
                    "score": 0.89,
                    "module": "module-0-foundations",
                    "chapter": "ch03-humanoid-landscape",
                }
            ]
        elif any(word in query_lower for word in ['ros', 'ros2']):
            answer = """ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides:

- Communication infrastructure (nodes, topics, services, actions)
- Hardware abstraction and device drivers
- Package management and build system
- Tools for visualization and debugging

ROS 2 improvements over ROS 1 include real-time capabilities, better security, and multi-platform support.

[This is a DEMO response - OpenAI integration is not active]"""
            sources = [
                {
                    "text": "ROS 2 is the next generation of the Robot Operating System, providing a middleware framework for robot development...",
                    "score": 0.93,
                    "module": "module-1-ros2",
                    "chapter": "ch04-ros2-setup",
                }
            ]
        else:
            answer = f"""Thank you for your question about: "{query}"

This is a DEMO MODE response. The chatbot is currently running without OpenAI integration.

To enable full AI-powered responses:
1. Add OpenAI API credits to your account
2. Ensure the API key in .env has sufficient quota
3. Restart the backend server

In demo mode, the chatbot can show you how the interface works, but responses are pre-programmed examples rather than AI-generated content from the Physical AI book.

[This is a DEMO response - OpenAI integration is not active]"""
            sources = [
                {
                    "text": "Demo source content - this would normally contain relevant excerpts from the Physical AI book...",
                    "score": 0.85,
                    "module": "demo-module",
                    "chapter": "demo-chapter",
                }
            ]

        return {
            "answer": answer,
            "sources": sources,
        }
