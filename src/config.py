"""
Configuration module for S.H.A.R.P.
Contains all global variables, model names, and settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for the SHARP pipeline."""
    
    # ============== Paths ==============
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    INDICES_DIR = BASE_DIR / "indices"
    
    # ============== API Keys ==============
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    # ============== Model Names ==============
    # LLM for generation
    LLM_MODEL = "models/gemini-flash-lite-latest"
    
    # Embedding model for dense retrieval
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    # Cross-encoder model for re-ranking
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # ============== Chunking Settings ==============
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # ============== Retrieval Settings ==============
    TOP_K = 10  # Number of documents to retrieve per method
    
    # ============== Hybrid Retrieval ==============
    DEFAULT_ALPHA = 0.5  # Weight for dense retrieval in hybrid mode
    
    # ============== Index File Names ==============
    BM25_INDEX_FILE = "bm25_index.pkl"
    FAISS_INDEX_FILE = "faiss_index.faiss"
    DOCUMENTS_FILE = "documents.pkl"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configurations are set."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in your .env file."
            )
        return True
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.INDICES_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_index_path(cls, filename: str) -> Path:
        """Get the full path for an index file."""
        return cls.INDICES_DIR / filename
