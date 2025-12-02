"""
S.H.A.R.P. - Scientific Hybrid Augmented Retrieval Pipeline

A domain-adaptive Hybrid RAG system designed to extract actionable steps 
from scientific PDFs using BM25, Dense Retrieval, and Cross-Encoder Re-ranking.
"""

from .config import Config
from .ingestion import DocumentProcessor
from .retrieval import SparseRetriever, DenseRetriever, HybridRetriever
from .reranker import CrossEncoderReranker
from .generation import GeminiGenerator

__version__ = "1.0.0"
__all__ = [
    "Config",
    "DocumentProcessor", 
    "SparseRetriever",
    "DenseRetriever",
    "HybridRetriever",
    "CrossEncoderReranker",
    "GeminiGenerator",
]
