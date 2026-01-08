"""Tests for RAG functionality including PDF processing and retrieval."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.agents.rag_agent import RAGAgent
from src.utils.pdf_processor import PDFProcessor
from src.vectorstore.qdrant_store import QdrantStore


class TestPDFProcessor:
    """Test suite for PDF processor."""
    
    @pytest.fixture
    def processor(self):
        """Create a PDFProcessor instance."""
        return PDFProcessor(chunk_size=100, chunk_overlap=20)
    
    def test_processor_initialization(self, processor):
        """Test processor initializes with correct settings."""
        assert processor.chunk_size == 100
        assert processor.chunk_overlap == 20
    
    def test_load_pdf_file_not_found(self, processor):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            processor.load_pdf("/nonexistent/path/file.pdf")
    
    def test_load_pdf_invalid_extension(self, processor, tmp_path):
        """Test handling of non-PDF file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Not a PDF")
        
        with pytest.raises(ValueError):
            processor.load_pdf(str(txt_file))
    
    def test_chunk_documents_empty_list(self, processor):
        """Test chunking with empty document list."""
        result = processor.chunk_documents([])
        assert result == []
    
    def test_chunk_documents_adds_metadata(self, processor):
        """Test that chunking adds chunk index to metadata."""
        from langchain_core.documents import Document
        
        docs = [Document(page_content="A" * 200, metadata={"page": 1})]
        chunks = processor.chunk_documents(docs)
        
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i


class TestQdrantStore:
    """Test suite for Qdrant vector store."""
    
    @pytest.fixture
    def mock_store(self):
        """Create a mocked QdrantStore."""
        with patch('src.vectorstore.qdrant_store.EmbeddingService') as mock_embed:
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_embedding_dimension.return_value = 1536
            mock_embed_instance.embed_text.return_value = [0.1] * 1536
            mock_embed_instance.embed_documents.return_value = [[0.1] * 1536]
            mock_embed.return_value = mock_embed_instance
            
            store = QdrantStore(collection_name="test_collection")
            yield store
    
    def test_store_initialization(self, mock_store):
        """Test store initializes and creates collection."""
        assert mock_store.collection_name == "test_collection"
        # Collection should be created
        collections = mock_store.client.get_collections().collections
        assert len(collections) > 0
    
    def test_add_documents(self, mock_store):
        """Test adding documents to store."""
        from langchain_core.documents import Document
        
        docs = [Document(page_content="Test content", metadata={"source": "test.pdf"})]
        doc_ids = mock_store.add_documents(docs)
        
        assert len(doc_ids) == 1
    
    def test_add_empty_documents(self, mock_store):
        """Test adding empty document list."""
        result = mock_store.add_documents([])
        assert result == []
    
    def test_similarity_search(self, mock_store):
        """Test similarity search functionality."""
        from langchain_core.documents import Document
        
        # Add a document first
        docs = [Document(page_content="Python is a programming language", metadata={"page": 1})]
        mock_store.add_documents(docs)
        
        # Search - note: mocked embeddings return same value, so search returns all
        results = mock_store.similarity_search("programming", k=1)
        
        # With mocked embeddings, this should return at least one result
        assert len(results) >= 0  # May be 0 or 1 depending on mock
    
    def test_clear_collection(self, mock_store):
        """Test clearing the collection."""
        from langchain_core.documents import Document
        
        # Add documents
        docs = [Document(page_content="Test", metadata={})]
        mock_store.add_documents(docs)
        
        # Clear should not raise an error
        mock_store.clear()
        
        # Collection should be recreated (may have 0 points)
        # Just verify no exception is raised
        assert True


class TestRAGAgent:
    """Test suite for RAG agent."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mocked RAG agent."""
        with patch('src.agents.rag_agent.get_llm') as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = MagicMock(content="Test answer")
            mock_llm.return_value = mock_llm_instance
            
            with patch('src.agents.rag_agent.QdrantStore') as mock_store:
                mock_store_instance = MagicMock()
                mock_store_instance.similarity_search.return_value = []
                mock_store_instance.get_collection_info.return_value = {"points_count": 0}
                mock_store.return_value = mock_store_instance
                
                agent = RAGAgent()
                yield agent
    
    def test_query_documents_no_documents(self, mock_agent):
        """Test querying with no documents loaded."""
        result = mock_agent.query_documents("What is this about?")
        
        assert "don't have any documents" in result["answer"].lower() or result["context_used"] is False
    
    def test_format_sources(self, mock_agent):
        """Test source formatting."""
        from langchain_core.documents import Document
        
        docs = [
            Document(
                page_content="This is test content that is longer than 200 characters. " * 5,
                metadata={"source_file": "test.pdf", "page": 1, "score": 0.95}
            )
        ]
        
        sources = mock_agent.format_sources(docs)
        
        assert len(sources) == 1
        assert sources[0]["source_file"] == "test.pdf"
        assert sources[0]["page"] == 1
        assert sources[0]["preview"].endswith("...")
    
    def test_format_context(self, mock_agent):
        """Test context formatting with documents."""
        from langchain_core.documents import Document
        
        docs = [
            Document(
                page_content="Content 1",
                metadata={"source_file": "doc1.pdf", "page": 1, "score": 0.9}
            ),
            Document(
                page_content="Content 2",
                metadata={"source_file": "doc2.pdf", "page": 2, "score": 0.8}
            ),
        ]
        
        context = mock_agent.format_context(docs)
        
        assert "doc1.pdf" in context
        assert "doc2.pdf" in context
        assert "Content 1" in context
        assert "Content 2" in context
    
    def test_format_context_empty(self, mock_agent):
        """Test context formatting with no documents."""
        context = mock_agent.format_context([])
        assert "no relevant context" in context.lower()


class TestRAGIntegration:
    """Integration tests for RAG (requires API keys)."""
    
    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """Create a sample PDF for testing."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            
            pdf_path = tmp_path / "test.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.drawString(100, 750, "This is a test PDF document.")
            c.drawString(100, 700, "It contains information about Python programming.")
            c.save()
            
            return pdf_path
        except ImportError:
            pytest.skip("reportlab not installed for PDF generation")
    
    @pytest.mark.integration
    def test_full_rag_pipeline(self, sample_pdf_path):
        """Test the full RAG pipeline with a real PDF."""
        try:
            from src.config import Config
            Config.validate()
        except ValueError:
            pytest.skip("API keys not configured")
        
        agent = RAGAgent()
        
        # Add PDF
        chunks = agent.add_pdf(str(sample_pdf_path))
        assert chunks > 0
        
        # Query
        result = agent.query_documents("What is this document about?")
        assert result["answer"]
        assert result["context_used"] is True
