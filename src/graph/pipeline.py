"""LangGraph pipeline for the AI agent system."""

from typing import TypedDict, Annotated, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.graph.router import QueryRouter
from src.agents.weather_agent import WeatherAgent
from src.agents.rag_agent import RAGAgent
from src.vectorstore.qdrant_store import QdrantStore


class AgentState(TypedDict):
    """State schema for the agent pipeline."""
    
    # Input
    query: str
    
    # Routing
    route: str
    route_reasoning: str
    
    # Response
    response: str
    sources: List[dict]
    
    # Metadata
    agent_used: str
    error: Optional[str]


class AgentPipeline:
    """LangGraph-based agent pipeline with weather and RAG capabilities."""
    
    def __init__(self, vector_store: Optional[QdrantStore] = None):
        """
        Initialize the agent pipeline.
        
        Args:
            vector_store: Optional shared QdrantStore instance
        """
        self.vector_store = vector_store or QdrantStore()
        self.router = QueryRouter()
        self.weather_agent = WeatherAgent()
        self.rag_agent = RAGAgent(vector_store=self.vector_store)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph workflow."""
        
        # Create the graph with our state schema
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("weather", self._weather_node)
        workflow.add_node("rag", self._rag_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "weather": "weather",
                "document": "rag",
            }
        )
        
        # Add edges to END
        workflow.add_edge("weather", END)
        workflow.add_edge("rag", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _router_node(self, state: AgentState) -> dict:
        """
        Router node that classifies the query.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with routing decision
        """
        try:
            decision = self.router.classify(state["query"])
            return {
                "route": decision.route,
                "route_reasoning": decision.reasoning,
            }
        except Exception as e:
            # Default to document if routing fails
            return {
                "route": "document",
                "route_reasoning": f"Routing failed: {str(e)}, defaulting to document",
                "error": str(e),
            }
    
    def _route_decision(self, state: AgentState) -> Literal["weather", "document"]:
        """
        Get the routing decision from state.
        
        Args:
            state: Current agent state
            
        Returns:
            Route string for conditional edge
        """
        return state["route"]
    
    def _weather_node(self, state: AgentState) -> dict:
        """
        Weather agent node.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with weather response
        """
        try:
            response = self.weather_agent.answer_query(state["query"])
            return {
                "response": response,
                "sources": [],
                "agent_used": "weather",
            }
        except Exception as e:
            return {
                "response": f"Sorry, I encountered an error getting the weather: {str(e)}",
                "sources": [],
                "agent_used": "weather",
                "error": str(e),
            }
    
    def _rag_node(self, state: AgentState) -> dict:
        """
        RAG agent node.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with RAG response
        """
        try:
            result = self.rag_agent.query_documents(state["query"])
            return {
                "response": result["answer"],
                "sources": result["sources"],
                "agent_used": "rag",
            }
        except Exception as e:
            return {
                "response": f"Sorry, I encountered an error searching the documents: {str(e)}",
                "sources": [],
                "agent_used": "rag",
                "error": str(e),
            }
    
    def invoke(self, query: str) -> AgentState:
        """
        Run the pipeline with a query.
        
        Args:
            query: User's query string
            
        Returns:
            Final agent state with response
        """
        initial_state: AgentState = {
            "query": query,
            "route": "",
            "route_reasoning": "",
            "response": "",
            "sources": [],
            "agent_used": "",
            "error": None,
        }
        
        result = self.graph.invoke(initial_state)
        return result
    
    def add_pdf(self, file_path: str) -> int:
        """
        Add a PDF to the RAG agent's vector store.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Number of chunks added
        """
        return self.rag_agent.add_pdf(file_path)
    
    def add_pdf_bytes(self, file_bytes: bytes, filename: str) -> int:
        """
        Add a PDF from bytes to the RAG agent's vector store.
        
        Args:
            file_bytes: PDF file content as bytes
            filename: Original filename for metadata
            
        Returns:
            Number of chunks added
        """
        return self.rag_agent.add_pdf_bytes(file_bytes, filename)
    
    def get_store_info(self) -> dict:
        """Get information about the vector store."""
        return self.rag_agent.get_store_info()
    
    def clear_documents(self):
        """Clear all documents from the vector store."""
        self.rag_agent.clear_documents()


def create_pipeline(vector_store: Optional[QdrantStore] = None) -> AgentPipeline:
    """
    Create and return an agent pipeline instance.
    
    Args:
        vector_store: Optional shared QdrantStore instance
        
    Returns:
        Configured AgentPipeline
    """
    return AgentPipeline(vector_store=vector_store)
