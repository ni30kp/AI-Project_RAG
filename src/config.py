"""Configuration and environment setup for the AI pipeline."""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    
    # LangSmith settings
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ai-pipeline-assignment")
    
    # Model settings
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Embedding model (using free HuggingFace embeddings)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector store settings
    QDRANT_COLLECTION_NAME = "pdf_documents"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required environment variables are set."""
        required = [
            ("GROQ_API_KEY", cls.GROQ_API_KEY),
            ("OPENWEATHERMAP_API_KEY", cls.OPENWEATHERMAP_API_KEY),
        ]
        
        missing = [name for name, value in required if not value]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return True


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Get configured LLM instance using Groq."""
    return ChatGroq(
        model=Config.GROQ_MODEL,
        temperature=temperature,
        api_key=Config.GROQ_API_KEY,
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get configured embeddings instance using HuggingFace (free, runs locally)."""
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
    )
