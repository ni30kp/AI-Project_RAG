"""Weather agent for fetching real-time weather data from OpenWeatherMap API."""

import requests
from typing import Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from src.config import Config, get_llm


class WeatherAgent:
    """Agent for handling weather-related queries using OpenWeatherMap API."""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self):
        """Initialize the weather agent."""
        self.api_key = Config.OPENWEATHERMAP_API_KEY
        self.llm = get_llm(temperature=0.3)
    
    def fetch_weather(self, city: str, units: str = "metric") -> dict:
        """
        Fetch weather data for a given city.
        
        Args:
            city: Name of the city to get weather for
            units: Temperature units ('metric', 'imperial', or 'kelvin')
            
        Returns:
            Dictionary containing weather data or error information
        """
        try:
            params = {
                "q": city,
                "appid": self.api_key,
                "units": units,
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            elif response.status_code == 404:
                return {"success": False, "error": f"City '{city}' not found"}
            else:
                return {
                    "success": False, 
                    "error": f"API error: {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}
    
    def format_weather_response(self, data: dict) -> str:
        """
        Format raw weather API response into a readable string.
        
        Args:
            data: Raw weather data from OpenWeatherMap API
            
        Returns:
            Formatted weather information string
        """
        if not data.get("success"):
            return f"Error: {data.get('error', 'Unknown error')}"
        
        weather = data["data"]
        
        # Extract relevant information
        city_name = weather.get("name", "Unknown")
        country = weather.get("sys", {}).get("country", "")
        temp = weather.get("main", {}).get("temp", "N/A")
        feels_like = weather.get("main", {}).get("feels_like", "N/A")
        humidity = weather.get("main", {}).get("humidity", "N/A")
        description = weather.get("weather", [{}])[0].get("description", "N/A")
        wind_speed = weather.get("wind", {}).get("speed", "N/A")
        
        formatted = f"""
**Weather in {city_name}, {country}**

🌡️ **Temperature:** {temp}°C (Feels like: {feels_like}°C)
💧 **Humidity:** {humidity}%
🌤️ **Conditions:** {description.title()}
💨 **Wind Speed:** {wind_speed} m/s
"""
        return formatted.strip()
    
    def extract_city_from_query(self, query: str) -> Optional[str]:
        """
        Use LLM to extract the city name from a natural language query.
        
        Args:
            query: User's natural language query
            
        Returns:
            Extracted city name or None if not found
        """
        prompt = f"""Extract the city name from this weather query. 
Return ONLY the city name, nothing else. 
If no city is mentioned, return "NONE".

Query: {query}

City name:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        city = response.content.strip()
        
        return None if city.upper() == "NONE" else city
    
    def answer_query(self, query: str) -> str:
        """
        Process a weather query and return a natural language response.
        
        Args:
            query: User's natural language weather query
            
        Returns:
            Natural language response with weather information
        """
        # Extract city from query
        city = self.extract_city_from_query(query)
        
        if not city:
            return "I couldn't identify a city in your query. Please specify a city name, e.g., 'What's the weather in London?'"
        
        # Fetch weather data
        weather_data = self.fetch_weather(city)
        
        if not weather_data.get("success"):
            return f"Sorry, I couldn't get the weather for {city}. {weather_data.get('error', '')}"
        
        # Format the weather data
        formatted_weather = self.format_weather_response(weather_data)
        
        # Generate a natural language response using LLM
        prompt = f"""Based on this weather data, provide a helpful and conversational response to the user's query.

User's query: {query}

Weather Data:
{formatted_weather}

Provide a friendly, informative response:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return response.content


# Create a LangChain tool for the weather agent
@tool
def get_weather(city: str) -> str:
    """
    Get current weather information for a city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        Weather information for the specified city
    """
    agent = WeatherAgent()
    weather_data = agent.fetch_weather(city)
    return agent.format_weather_response(weather_data)
