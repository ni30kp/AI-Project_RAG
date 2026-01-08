"""Embedding service for generating vector representations of text."""

from typing import List
from langchain_core.documents import Document

from src.config import get_embeddings


class EmbeddingService:
    """Service for generating embeddings using OpenAI models."""
    
    def __init__(self):
        """Initialize the embedding service."""
        self.embeddings = get_embeddings()
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple text strings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for a list of Document objects.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of embedding vectors
        """
        texts = [doc.page_content for doc in documents]
        return self.embed_texts(texts)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Integer dimension of embeddings
        """
        # Generate a test embedding to get dimension
        test_embedding = self.embed_text("test")
        return len(test_embedding)
