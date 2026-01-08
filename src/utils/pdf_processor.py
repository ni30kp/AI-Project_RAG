"""PDF processor for loading and chunking PDF documents."""

from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import Config


class PDFProcessor:
    """Handles PDF loading and text chunking for RAG pipeline."""
    
    def __init__(
        self, 
        chunk_size: int = None, 
        chunk_overlap: int = None
    ):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Size of text chunks (default from Config)
            chunk_overlap: Overlap between chunks (default from Config)
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and return its contents as documents.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects with page content and metadata
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the file is not a PDF
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")
        
        loader = PyPDFLoader(str(path))
        documents = loader.load()
        
        # Add source filename to metadata
        for doc in documents:
            doc.metadata["source_file"] = path.name
        
        return documents
    
    def load_pdf_from_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """
        Load a PDF from bytes (useful for Streamlit file uploads).
        
        Args:
            file_bytes: PDF file content as bytes
            filename: Original filename for metadata
            
        Returns:
            List of Document objects
        """
        import tempfile
        import os
        
        # Write bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        
        try:
            documents = self.load_pdf(tmp_path)
            # Update metadata with original filename
            for doc in documents:
                doc.metadata["source_file"] = filename
            return documents
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for embedding.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects with preserved metadata
        """
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        
        return chunks
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Load and chunk a PDF file in one step.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of chunked Document objects ready for embedding
        """
        documents = self.load_pdf(file_path)
        return self.chunk_documents(documents)
    
    def process_pdf_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """
        Load and chunk PDF from bytes in one step.
        
        Args:
            file_bytes: PDF file content as bytes
            filename: Original filename for metadata
            
        Returns:
            List of chunked Document objects ready for embedding
        """
        documents = self.load_pdf_from_bytes(file_bytes, filename)
        return self.chunk_documents(documents)
