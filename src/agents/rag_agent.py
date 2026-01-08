"""RAG agent for answering questions from PDF documents."""

from typing import List, Optional
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_llm
from src.vectorstore.qdrant_store import QdrantStore
from src.utils.pdf_processor import PDFProcessor


class RAGAgent:
    """Agent for handling document-based questions using RAG."""
    
    def __init__(self, vector_store: Optional[QdrantStore] = None):
        """
        Initialize the RAG agent.
        
        Args:
            vector_store: Optional pre-configured QdrantStore instance
        """
        self.vector_store = vector_store or QdrantStore()
        self.pdf_processor = PDFProcessor()
        self.llm = get_llm(temperature=0.2)
        
        # RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
            
Rules:
1. Only use information from the provided context to answer questions
2. If the context doesn't contain relevant information, say so clearly
3. Cite the source when possible (page numbers, document names)
4. Be concise but thorough in your answers
5. If you're unsure, express uncertainty rather than making things up"""),
            ("human", """Context:
{context}

Question: {question}

Please answer the question based on the context provided above."""),
        ])
    
    def add_pdf(self, file_path: str) -> int:
        """
        Process and add a PDF to the vector store.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Number of chunks added to the store
        """
        chunks = self.pdf_processor.process_pdf(file_path)
        self.vector_store.add_documents(chunks)
        return len(chunks)
    
    def add_pdf_bytes(self, file_bytes: bytes, filename: str) -> int:
        """
        Process and add a PDF from bytes to the vector store.
        
        Args:
            file_bytes: PDF file content as bytes
            filename: Original filename for metadata
            
        Returns:
            Number of chunks added to the store
        """
        chunks = self.pdf_processor.process_pdf_bytes(file_bytes, filename)
        self.vector_store.add_documents(chunks)
        return len(chunks)
    
    def retrieve_context(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Question to find relevant context for
            k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Formatted context string with source citations
        """
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "N/A")
            score = doc.metadata.get("score", 0)
            
            context_parts.append(
                f"[Source {i}: {source}, Page {page}, Relevance: {score:.2f}]\n{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def format_sources(self, documents: List[Document]) -> List[dict]:
        """
        Format source information from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of source dictionaries with metadata
        """
        sources = []
        for doc in documents:
            sources.append({
                "source_file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "chunk_index": doc.metadata.get("chunk_index", "N/A"),
                "relevance_score": doc.metadata.get("score", 0),
                "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })
        return sources
    
    def query_documents(self, question: str, k: int = 4) -> dict:
        """
        Answer a question using RAG.
        
        Args:
            question: Question to answer
            k: Number of documents to retrieve for context
            
        Returns:
            Dictionary with answer and sources
        """
        # Retrieve relevant documents
        documents = self.retrieve_context(question, k=k)
        
        if not documents:
            return {
                "answer": "I don't have any documents to search. Please upload a PDF first.",
                "sources": [],
                "context_used": False,
            }
        
        # Format context
        context = self.format_context(documents)
        
        # Generate answer using LLM
        messages = self.rag_prompt.format_messages(
            context=context,
            question=question,
        )
        
        response = self.llm.invoke(messages)
        
        return {
            "answer": response.content,
            "sources": self.format_sources(documents),
            "context_used": True,
        }
    
    def get_store_info(self) -> dict:
        """
        Get information about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        return self.vector_store.get_collection_info()
    
    def clear_documents(self):
        """Clear all documents from the vector store."""
        self.vector_store.clear()
