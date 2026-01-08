"""Streamlit UI for the AI Pipeline application."""

import streamlit as st
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.graph.pipeline import create_pipeline


# Page configuration
st.set_page_config(
    page_title="AI Pipeline - Weather & Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    
    .source-card {
        background-color: #fff3e0;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
    
    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .weather-badge {
        background-color: #bbdefb;
        color: #1565c0;
    }
    
    .rag-badge {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "pipeline" not in st.session_state:
        try:
            Config.validate()
            st.session_state.pipeline = create_pipeline()
            st.session_state.initialized = True
        except ValueError as e:
            st.session_state.initialized = False
            st.session_state.init_error = str(e)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = 0


def render_sidebar():
    """Render the sidebar with PDF upload and settings."""
    with st.sidebar:
        st.header("📄 Document Upload")
        
        # PDF upload
        uploaded_file = st.file_uploader(
            "Upload a PDF to enable document Q&A",
            type=["pdf"],
            help="Upload a PDF file to ask questions about its content"
        )
        
        if uploaded_file is not None:
            if st.button("📥 Process PDF", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    try:
                        chunks_added = st.session_state.pipeline.add_pdf_bytes(
                            uploaded_file.read(),
                            uploaded_file.name
                        )
                        st.session_state.documents_loaded += chunks_added
                        st.success(f"✅ Added {chunks_added} chunks from {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
        
        # Document stats
        st.divider()
        st.subheader("📊 Statistics")
        
        if st.session_state.initialized:
            try:
                info = st.session_state.pipeline.get_store_info()
                st.metric("Documents in Store", info.get("points_count", 0))
            except Exception:
                st.metric("Documents Loaded", st.session_state.documents_loaded)
        
        # Clear documents
        st.divider()
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            if st.session_state.initialized:
                st.session_state.pipeline.clear_documents()
                st.session_state.documents_loaded = 0
                st.success("Documents cleared!")
        
        # Clear chat
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Info section
        st.divider()
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        **Weather Queries:**
        - "What's the weather in London?"
        - "Is it raining in Tokyo?"
        - "Temperature in New York"
        
        **Document Queries:**
        - Upload a PDF first
        - "Summarize the document"
        - "What does it say about X?"
        """)


def render_message(message: dict):
    """Render a chat message."""
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    else:
        with st.chat_message("assistant"):
            # Show agent badge
            agent_used = message.get("agent_used", "")
            if agent_used == "weather":
                st.markdown('<span class="agent-badge weather-badge">🌤️ Weather Agent</span>', unsafe_allow_html=True)
            elif agent_used == "rag":
                st.markdown('<span class="agent-badge rag-badge">📄 Document Agent</span>', unsafe_allow_html=True)
            
            st.write(content)
            
            # Show sources if available
            sources = message.get("sources", [])
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} documents)"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"""
                        **Source {i}:** {source.get('source_file', 'Unknown')} (Page {source.get('page', 'N/A')})
                        
                        *Relevance: {source.get('relevance_score', 0):.2f}*
                        
                        > {source.get('preview', '')}
                        """)
                        if i < len(sources):
                            st.divider()


def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Header
    st.title("🤖 AI Pipeline Assistant")
    st.caption("Ask about the weather or query your documents using AI")
    
    # Check initialization
    if not st.session_state.get("initialized", False):
        st.error(f"⚠️ Configuration Error: {st.session_state.get('init_error', 'Unknown error')}")
        st.info("Please check your .env file and ensure all required API keys are set.")
        st.code("""
# Required environment variables:
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key

# Optional (for LangSmith tracing):
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-pipeline-assignment
        """)
        return
    
    # Render sidebar
    render_sidebar()
    
    # Chat interface
    st.divider()
    
    # Display chat history
    for message in st.session_state.messages:
        render_message(message)
    
    # Chat input
    if prompt := st.chat_input("Ask about weather or your documents..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from pipeline
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.pipeline.invoke(prompt)
                    
                    # Show agent badge
                    agent_used = result.get("agent_used", "")
                    if agent_used == "weather":
                        st.markdown('<span class="agent-badge weather-badge">🌤️ Weather Agent</span>', unsafe_allow_html=True)
                    elif agent_used == "rag":
                        st.markdown('<span class="agent-badge rag-badge">📄 Document Agent</span>', unsafe_allow_html=True)
                    
                    # Show response
                    st.write(result["response"])
                    
                    # Show sources if available
                    sources = result.get("sources", [])
                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)} documents)"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"""
                                **Source {i}:** {source.get('source_file', 'Unknown')} (Page {source.get('page', 'N/A')})
                                
                                *Relevance: {source.get('relevance_score', 0):.2f}*
                                
                                > {source.get('preview', '')}
                                """)
                                if i < len(sources):
                                    st.divider()
                    
                    # Store message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "agent_used": agent_used,
                        "sources": sources,
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


if __name__ == "__main__":
    main()
