"""Router node for classifying queries and determining the appropriate agent."""

from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import get_llm


class RouteDecision(BaseModel):
    """Schema for routing decision."""
    
    route: Literal["weather", "document"] = Field(
        description="The route to take: 'weather' for weather-related queries, 'document' for document/PDF questions"
    )
    reasoning: str = Field(
        description="Brief explanation of why this route was chosen"
    )


class QueryRouter:
    """Router that classifies queries and determines which agent to use."""
    
    def __init__(self):
        """Initialize the query router."""
        self.llm = get_llm(temperature=0)
        
        # Use structured output for reliable classification
        self.structured_llm = self.llm.with_structured_output(RouteDecision)
        
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier. Your job is to determine whether a user's query is:

1. **weather**: Questions about current weather, temperature, forecasts, or weather conditions in any location.
   Examples: "What's the weather in Paris?", "Is it raining in Tokyo?", "Temperature in New York"

2. **document**: Questions about information in uploaded documents/PDFs, or questions that require searching through document content.
   Examples: "What does the document say about X?", "Summarize the PDF", "Find information about Y in the uploaded file"

Choose the most appropriate route based on the query intent."""),
            ("human", "Query: {query}\n\nClassify this query and explain your reasoning:"),
        ])
    
    def classify(self, query: str) -> RouteDecision:
        """
        Classify a query and return the routing decision.
        
        Args:
            query: User's query string
            
        Returns:
            RouteDecision with route and reasoning
        """
        messages = self.classification_prompt.format_messages(query=query)
        decision = self.structured_llm.invoke(messages)
        return decision
    
    def route(self, query: str) -> str:
        """
        Get the route for a query.
        
        Args:
            query: User's query string
            
        Returns:
            Route string: 'weather' or 'document'
        """
        decision = self.classify(query)
        return decision.route


def route_query(query: str) -> str:
    """
    Convenience function to route a query.
    
    Args:
        query: User's query string
        
    Returns:
        Route string: 'weather' or 'document'
    """
    router = QueryRouter()
    return router.route(query)
