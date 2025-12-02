"""
Ingestion module for S.H.A.R.P.
Handles PDF loading, text cleaning, and chunking with metadata preservation.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import Config


class DocumentProcessor:
    """
    Handles the complete ingestion pipeline:
    1. Load PDFs from the data directory
    2. Clean extracted text
    3. Split into chunks with metadata
    """
    
    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of each text chunk (default: 500)
            chunk_overlap: Overlap between chunks (default: 50)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted PDF text.
        
        Removes:
        - Citation markers like [1], [2,3], [1-5]
        - Excessive whitespace and newlines
        - Special characters that don't add meaning
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text string
        """
        # Remove citation markers like [1], [2,3], [1-5], etc.
        text = re.sub(r'\[\d+(?:[-,]\d+)*\]', '', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def load_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Load a single PDF file and return documents with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects with page content and metadata
        """
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        # Add source filename and clean text
        for page in pages:
            page.page_content = self.clean_text(page.page_content)
            page.metadata["source"] = pdf_path.name
            # Ensure page number is present (PyPDFLoader uses 0-indexed pages)
            if "page" not in page.metadata:
                page.metadata["page"] = page.metadata.get("page_number", 0)
        
        return pages
    
    def load_all_pdfs(self, data_dir: Optional[Path] = None) -> List[Document]:
        """
        Load all PDFs from the data directory.
        
        Args:
            data_dir: Directory containing PDFs (default: Config.DATA_DIR)
            
        Returns:
            List of all Document objects from all PDFs
        """
        if data_dir is None:
            data_dir = Config.DATA_DIR
        
        all_documents = []
        pdf_files = list(data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {data_dir}")
            return all_documents
        
        for pdf_path in pdf_files:
            try:
                docs = self.load_pdf(pdf_path)
                all_documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {pdf_path.name}")
            except Exception as e:
                print(f"Error loading {pdf_path.name}: {e}")
        
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks while preserving metadata.
        
        Args:
            documents: List of Document objects (typically one per page)
            
        Returns:
            List of chunked Document objects with preserved metadata
        """
        chunked_docs = []
        
        for doc in documents:
            # Split the document content
            chunks = self.text_splitter.split_text(doc.page_content)
            
            # Create new documents for each chunk with preserved metadata
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                )
                chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def process(self, data_dir: Optional[Path] = None) -> List[Document]:
        """
        Complete processing pipeline: Load → Clean → Chunk.
        
        Args:
            data_dir: Directory containing PDFs (default: Config.DATA_DIR)
            
        Returns:
            List of processed and chunked Document objects
        """
        # Load all PDFs
        documents = self.load_all_pdfs(data_dir)
        
        if not documents:
            return []
        
        # Chunk the documents
        chunked_docs = self.chunk_documents(documents)
        
        print(f"Processed {len(documents)} pages into {len(chunked_docs)} chunks")
        
        return chunked_docs
    
    def process_uploaded_file(self, file_path: Path) -> List[Document]:
        """
        Process a single uploaded PDF file.
        
        Args:
            file_path: Path to the uploaded PDF
            
        Returns:
            List of processed and chunked Document objects
        """
        documents = self.load_pdf(file_path)
        chunked_docs = self.chunk_documents(documents)
        
        print(f"Processed {len(documents)} pages into {len(chunked_docs)} chunks")
        
        return chunked_docs
