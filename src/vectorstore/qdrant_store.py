"""Qdrant vector store for document storage and retrieval."""

from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from langchain_core.documents import Document
import uuid

from src.config import Config
from src.vectorstore.embeddings import EmbeddingService


class QdrantStore:
    """Qdrant vector store for storing and retrieving document embeddings."""
    
    def __init__(self, collection_name: str = None):
        """
        Initialize the Qdrant store.
        
        Args:
            collection_name: Name of the collection (default from Config)
        """
        self.collection_name = collection_name or Config.QDRANT_COLLECTION_NAME
        self.embedding_service = EmbeddingService()
        
        # Initialize in-memory Qdrant client
        self.client = QdrantClient(":memory:")
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            # Get embedding dimension
            dimension = self.embedding_service.get_embedding_dimension()
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Generate embeddings
        embeddings = self.embedding_service.embed_documents(documents)
        
        # Create points with unique IDs
        points = []
        doc_ids = []
        
        for doc, embedding in zip(documents, embeddings):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            points.append(
                PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
            )
        
        # Upsert points to collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        return doc_ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter_metadata: Optional[dict] = None,
    ) -> List[Document]:
        """
        Search for similar documents based on a query.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of similar Document objects with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Build filter if provided
        query_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value),
                )
                for key, value in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)
        
        # Search using query_points method (compatible with qdrant-client 1.8.0+)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
            query_filter=query_filter,
        )
        results = search_result.points
        
        # Convert to Documents
        documents = []
        for result in results:
            doc = Document(
                page_content=result.payload["content"],
                metadata={
                    **result.payload["metadata"],
                    "score": result.score,
                },
            )
            documents.append(doc)
        
        return documents
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(collection_name=self.collection_name)
    
    def get_collection_info(self) -> dict:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        info = self.client.get_collection(collection_name=self.collection_name)
        # Handle both old and new Qdrant client API
        points_count = getattr(info, 'points_count', None) or getattr(info, 'vectors_count', 0)
        return {
            "name": self.collection_name,
            "points_count": points_count,
        }
    
    def clear(self):
        """Clear all documents from the collection."""
        self.delete_collection()
        self._ensure_collection()
