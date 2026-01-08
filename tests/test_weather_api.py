"""Tests for the Weather API integration."""

import pytest
from unittest.mock import patch, MagicMock
import requests

from src.agents.weather_agent import WeatherAgent


class TestWeatherAgent:
    """Test suite for WeatherAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create a WeatherAgent instance for testing."""
        with patch('src.agents.weather_agent.Config') as mock_config:
            mock_config.OPENWEATHERMAP_API_KEY = "test_api_key"
            with patch('src.agents.weather_agent.get_llm'):
                yield WeatherAgent()
    
    def test_fetch_weather_success(self, agent):
        """Test successful weather fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {
                "temp": 15.5,
                "feels_like": 14.0,
                "humidity": 80
            },
            "weather": [{"description": "overcast clouds"}],
            "wind": {"speed": 5.2}
        }
        
        with patch('requests.get', return_value=mock_response):
            result = agent.fetch_weather("London")
        
        assert result["success"] is True
        assert result["data"]["name"] == "London"
        assert result["data"]["main"]["temp"] == 15.5
    
    def test_fetch_weather_city_not_found(self, agent):
        """Test weather fetch for non-existent city."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        with patch('requests.get', return_value=mock_response):
            result = agent.fetch_weather("NonExistentCity123")
        
        assert result["success"] is False
        assert "not found" in result["error"].lower()
    
    def test_fetch_weather_timeout(self, agent):
        """Test handling of request timeout."""
        with patch('requests.get', side_effect=requests.exceptions.Timeout):
            result = agent.fetch_weather("London")
        
        assert result["success"] is False
        assert "timed out" in result["error"].lower()
    
    def test_fetch_weather_request_exception(self, agent):
        """Test handling of general request exceptions."""
        with patch('requests.get', side_effect=requests.exceptions.RequestException("Network error")):
            result = agent.fetch_weather("London")
        
        assert result["success"] is False
        assert "failed" in result["error"].lower()
    
    def test_format_weather_response_success(self, agent):
        """Test formatting of successful weather response."""
        weather_data = {
            "success": True,
            "data": {
                "name": "Paris",
                "sys": {"country": "FR"},
                "main": {
                    "temp": 22.0,
                    "feels_like": 21.5,
                    "humidity": 65
                },
                "weather": [{"description": "clear sky"}],
                "wind": {"speed": 3.5}
            }
        }
        
        result = agent.format_weather_response(weather_data)
        
        assert "Paris" in result
        assert "22" in result
        assert "Clear Sky" in result
        assert "65%" in result
    
    def test_format_weather_response_error(self, agent):
        """Test formatting of error response."""
        error_data = {
            "success": False,
            "error": "City not found"
        }
        
        result = agent.format_weather_response(error_data)
        
        assert "Error" in result
        assert "not found" in result.lower()
    
    def test_extract_city_from_query(self, agent):
        """Test city extraction from natural language query."""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = "London"
        agent.llm.invoke = MagicMock(return_value=mock_response)
        
        result = agent.extract_city_from_query("What's the weather in London?")
        
        assert result == "London"
    
    def test_extract_city_no_city_found(self, agent):
        """Test city extraction when no city is mentioned."""
        mock_response = MagicMock()
        mock_response.content = "NONE"
        agent.llm.invoke = MagicMock(return_value=mock_response)
        
        result = agent.extract_city_from_query("How are you?")
        
        assert result is None


class TestWeatherAPIIntegration:
    """Integration tests for weather API (requires API key)."""
    
    @pytest.fixture
    def live_agent(self):
        """Create a live WeatherAgent (requires real API key)."""
        try:
            from src.config import Config
            Config.validate()
            return WeatherAgent()
        except ValueError:
            pytest.skip("API keys not configured")
    
    @pytest.mark.integration
    def test_live_weather_fetch(self, live_agent):
        """Test actual API call to OpenWeatherMap."""
        result = live_agent.fetch_weather("London")
        
        assert result["success"] is True
        assert "name" in result["data"]
        assert "main" in result["data"]
