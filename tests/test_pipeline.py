"""Tests for the LangGraph pipeline."""

import pytest
from unittest.mock import patch, MagicMock

from src.graph.router import QueryRouter, RouteDecision
from src.graph.pipeline import AgentPipeline, create_pipeline


class TestQueryRouter:
    """Test suite for query router."""
    
    @pytest.fixture
    def mock_router(self):
        """Create a mocked router."""
        with patch('src.graph.router.get_llm') as mock_llm:
            mock_llm_instance = MagicMock()
            
            # Setup structured output mock
            structured_mock = MagicMock()
            mock_llm_instance.with_structured_output.return_value = structured_mock
            mock_llm.return_value = mock_llm_instance
            
            router = QueryRouter()
            # Store the mock for manipulation in tests
            router._structured_mock = structured_mock
            yield router
    
    def test_classify_weather_query(self, mock_router):
        """Test classification of weather query."""
        mock_router._structured_mock.invoke.return_value = RouteDecision(
            route="weather",
            reasoning="Query is about weather conditions"
        )
        
        decision = mock_router.classify("What's the weather in Paris?")
        
        assert decision.route == "weather"
    
    def test_classify_document_query(self, mock_router):
        """Test classification of document query."""
        mock_router._structured_mock.invoke.return_value = RouteDecision(
            route="document",
            reasoning="Query is about document content"
        )
        
        decision = mock_router.classify("What does the PDF say about machine learning?")
        
        assert decision.route == "document"
    
    def test_route_returns_string(self, mock_router):
        """Test that route() returns just the route string."""
        mock_router._structured_mock.invoke.return_value = RouteDecision(
            route="weather",
            reasoning="Weather query"
        )
        
        result = mock_router.route("Temperature in London?")
        
        assert result == "weather"
        assert isinstance(result, str)


class TestAgentPipeline:
    """Test suite for agent pipeline."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mocked pipeline."""
        with patch('src.graph.pipeline.QueryRouter') as mock_router, \
             patch('src.graph.pipeline.WeatherAgent') as mock_weather, \
             patch('src.graph.pipeline.RAGAgent') as mock_rag, \
             patch('src.graph.pipeline.QdrantStore') as mock_store:
            
            # Setup router mock
            mock_router_instance = MagicMock()
            mock_router_instance.classify.return_value = RouteDecision(
                route="weather",
                reasoning="Test"
            )
            mock_router.return_value = mock_router_instance
            
            # Setup weather agent mock
            mock_weather_instance = MagicMock()
            mock_weather_instance.answer_query.return_value = "Sunny, 25°C"
            mock_weather.return_value = mock_weather_instance
            
            # Setup RAG agent mock
            mock_rag_instance = MagicMock()
            mock_rag_instance.query_documents.return_value = {
                "answer": "Document answer",
                "sources": [],
                "context_used": True
            }
            mock_rag.return_value = mock_rag_instance
            
            # Setup store mock
            mock_store_instance = MagicMock()
            mock_store_instance.get_collection_info.return_value = {"points_count": 0}
            mock_store.return_value = mock_store_instance
            
            pipeline = AgentPipeline()
            pipeline._mock_router = mock_router_instance
            pipeline._mock_weather = mock_weather_instance
            pipeline._mock_rag = mock_rag_instance
            
            yield pipeline
    
    def test_pipeline_routes_to_weather(self, mock_pipeline):
        """Test pipeline routes weather queries correctly."""
        mock_pipeline._mock_router.classify.return_value = RouteDecision(
            route="weather",
            reasoning="Weather query"
        )
        
        result = mock_pipeline.invoke("What's the weather in Tokyo?")
        
        assert result["agent_used"] == "weather"
        assert "25°C" in result["response"]
    
    def test_pipeline_routes_to_rag(self, mock_pipeline):
        """Test pipeline routes document queries correctly."""
        mock_pipeline._mock_router.classify.return_value = RouteDecision(
            route="document",
            reasoning="Document query"
        )
        
        result = mock_pipeline.invoke("Summarize the PDF")
        
        assert result["agent_used"] == "rag"
        assert "Document answer" in result["response"]
    
    def test_pipeline_includes_sources_for_rag(self, mock_pipeline):
        """Test that RAG responses include sources."""
        mock_pipeline._mock_router.classify.return_value = RouteDecision(
            route="document",
            reasoning="Document query"
        )
        mock_pipeline._mock_rag.query_documents.return_value = {
            "answer": "Answer with sources",
            "sources": [{"source_file": "test.pdf", "page": 1}],
            "context_used": True
        }
        
        result = mock_pipeline.invoke("What is in the document?")
        
        assert len(result["sources"]) == 1
        assert result["sources"][0]["source_file"] == "test.pdf"
    
    def test_pipeline_handles_weather_error(self, mock_pipeline):
        """Test pipeline handles weather agent errors gracefully."""
        mock_pipeline._mock_router.classify.return_value = RouteDecision(
            route="weather",
            reasoning="Weather query"
        )
        mock_pipeline.weather_agent.answer_query.side_effect = Exception("API Error")
        
        result = mock_pipeline.invoke("Weather in XYZ?")
        
        assert "error" in result["response"].lower()
    
    def test_pipeline_handles_rag_error(self, mock_pipeline):
        """Test pipeline handles RAG agent errors gracefully."""
        mock_pipeline._mock_router.classify.return_value = RouteDecision(
            route="document",
            reasoning="Document query"
        )
        mock_pipeline.rag_agent.query_documents.side_effect = Exception("Retrieval Error")
        
        result = mock_pipeline.invoke("Find info in document")
        
        assert "error" in result["response"].lower()


class TestCreatePipeline:
    """Test the pipeline factory function."""
    
    def test_create_pipeline_returns_pipeline(self):
        """Test that create_pipeline returns an AgentPipeline."""
        with patch('src.graph.pipeline.QueryRouter'), \
             patch('src.graph.pipeline.WeatherAgent'), \
             patch('src.graph.pipeline.RAGAgent'), \
             patch('src.graph.pipeline.QdrantStore'):
            
            pipeline = create_pipeline()
            
            assert isinstance(pipeline, AgentPipeline)
    
    def test_create_pipeline_accepts_custom_store(self):
        """Test that create_pipeline accepts a custom vector store."""
        with patch('src.graph.pipeline.QueryRouter'), \
             patch('src.graph.pipeline.WeatherAgent'), \
             patch('src.graph.pipeline.RAGAgent'), \
             patch('src.graph.pipeline.QdrantStore'):
            
            custom_store = MagicMock()
            pipeline = create_pipeline(vector_store=custom_store)
            
            assert pipeline.vector_store == custom_store


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.integration
    def test_full_pipeline_weather(self):
        """Test full pipeline with real weather query."""
        try:
            from src.config import Config
            Config.validate()
        except ValueError:
            pytest.skip("API keys not configured")
        
        pipeline = create_pipeline()
        result = pipeline.invoke("What is the weather in London?")
        
        assert result["agent_used"] == "weather"
        assert result["response"]
    
    @pytest.mark.integration
    def test_full_pipeline_document_no_docs(self):
        """Test full pipeline with document query but no documents."""
        try:
            from src.config import Config
            Config.validate()
        except ValueError:
            pytest.skip("API keys not configured")
        
        pipeline = create_pipeline()
        result = pipeline.invoke("What does the document say?")
        
        assert result["agent_used"] == "rag"
        assert "upload" in result["response"].lower() or "don't have" in result["response"].lower()
