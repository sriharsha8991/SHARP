"""
Retrieval module for S.H.A.R.P.
Implements Sparse (BM25), Dense (FAISS), and Hybrid retrieval strategies.
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from langchain_core.documents import Document

from .config import Config


class SparseRetriever:
    """
    Sparse retrieval using BM25 algorithm.
    Tokenizes documents and builds an inverted index for keyword-based search.
    """
    
    def __init__(self):
        """Initialize the sparse retriever."""
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        self.tokenized_corpus: List[List[str]] = []
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Simple tokenization: lowercase and split on whitespace.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        return text.lower().split()
    
    def index(self, documents: List[Document]) -> None:
        """
        Build the BM25 index from documents.
        
        Args:
            documents: List of Document objects to index
        """
        self.documents = documents
        self.tokenized_corpus = [
            self.tokenize(doc.page_content) for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 index built with {len(documents)} documents")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = Config.TOP_K
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call index() first.")
        
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (self.documents[i], float(scores[i])) 
            for i in top_indices
        ]
        
        return results
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save the BM25 index to disk."""
        if path is None:
            path = Config.get_index_path(Config.BM25_INDEX_FILE)
        
        data = {
            "bm25": self.bm25,
            "documents": self.documents,
            "tokenized_corpus": self.tokenized_corpus
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"BM25 index saved to {path}")
    
    def load(self, path: Optional[Path] = None) -> None:
        """Load the BM25 index from disk."""
        if path is None:
            path = Config.get_index_path(Config.BM25_INDEX_FILE)
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.bm25 = data["bm25"]
        self.documents = data["documents"]
        self.tokenized_corpus = data["tokenized_corpus"]
        print(f"BM25 index loaded from {path}")


class DenseRetriever:
    """
    Dense retrieval using sentence embeddings and FAISS.
    Uses all-mpnet-base-v2 for semantic similarity search.
    """
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        """
        Initialize the dense retriever.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Build the FAISS index from documents.
        
        Args:
            documents: List of Document objects to index
        """
        self.documents = documents
        
        # Generate embeddings for all documents
        texts = [doc.page_content for doc in documents]
        self.embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Build FAISS index (Inner Product = Cosine Similarity after normalization)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        
        print(f"FAISS index built with {len(documents)} documents")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = Config.TOP_K
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        if self.index is None:
            raise ValueError("Index not built. Call index_documents() first.")
        
        # Encode and normalize query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = [
            (self.documents[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
        ]
        
        return results
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save the FAISS index and documents to disk."""
        if path is None:
            path = Config.get_index_path(Config.FAISS_INDEX_FILE)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path))
        
        # Save documents separately
        docs_path = Config.get_index_path(Config.DOCUMENTS_FILE)
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
        
        print(f"FAISS index saved to {path}")
    
    def load(self, path: Optional[Path] = None) -> None:
        """Load the FAISS index and documents from disk."""
        if path is None:
            path = Config.get_index_path(Config.FAISS_INDEX_FILE)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path))
        
        # Load documents
        docs_path = Config.get_index_path(Config.DOCUMENTS_FILE)
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
        
        print(f"FAISS index loaded from {path}")


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and Dense retrieval.
    Uses weighted fusion with min-max normalization.
    """
    
    def __init__(
        self,
        sparse_retriever: Optional[SparseRetriever] = None,
        dense_retriever: Optional[DenseRetriever] = None
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            sparse_retriever: Pre-initialized SparseRetriever
            dense_retriever: Pre-initialized DenseRetriever
        """
        self.sparse = sparse_retriever or SparseRetriever()
        self.dense = dense_retriever or DenseRetriever()
    
    def index(self, documents: List[Document]) -> None:
        """
        Build both sparse and dense indices.
        
        Args:
            documents: List of Document objects to index
        """
        self.sparse.index(documents)
        self.dense.index_documents(documents)
    
    @staticmethod
    def min_max_normalize(scores: List[float]) -> List[float]:
        """
        Apply min-max normalization to a list of scores.
        
        Args:
            scores: List of raw scores
            
        Returns:
            Normalized scores in [0, 1] range
        """
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def retrieve(
        self, 
        query: str, 
        alpha: float = Config.DEFAULT_ALPHA,
        top_k: int = Config.TOP_K
    ) -> List[Tuple[Document, float, Dict[str, float]]]:
        """
        Retrieve documents using weighted hybrid fusion.
        
        Formula: Score_final = (alpha × Score_dense) + ((1-alpha) × Score_sparse)
        
        Args:
            query: Search query
            alpha: Weight for dense retrieval (0-1)
            top_k: Number of documents to retrieve
            
        Returns:
            List of (Document, final_score, score_breakdown) tuples
        """
        # Get results from both retrievers (get more than top_k for better fusion)
        fetch_k = min(top_k * 2, len(self.sparse.documents))
        
        sparse_results = self.sparse.retrieve(query, fetch_k)
        dense_results = self.dense.retrieve(query, fetch_k)
        
        # Create score dictionaries keyed by document content hash
        sparse_scores = {
            hash(doc.page_content): score 
            for doc, score in sparse_results
        }
        dense_scores = {
            hash(doc.page_content): score 
            for doc, score in dense_results
        }
        
        # Get all unique documents
        all_docs = {}
        for doc, _ in sparse_results + dense_results:
            doc_hash = hash(doc.page_content)
            if doc_hash not in all_docs:
                all_docs[doc_hash] = doc
        
        # Normalize BM25 scores (they can range from 0 to 50+)
        all_sparse = [sparse_scores.get(h, 0.0) for h in all_docs.keys()]
        normalized_sparse = self.min_max_normalize(all_sparse)
        
        # Dense scores are already in [0, 1] range (cosine similarity)
        # but we normalize for consistency
        all_dense = [dense_scores.get(h, 0.0) for h in all_docs.keys()]
        normalized_dense = self.min_max_normalize(all_dense)
        
        # Calculate fused scores
        fused_results = []
        for i, (doc_hash, doc) in enumerate(all_docs.items()):
            sparse_norm = normalized_sparse[i]
            dense_norm = normalized_dense[i]
            
            # Weighted fusion formula
            final_score = (alpha * dense_norm) + ((1 - alpha) * sparse_norm)
            
            score_breakdown = {
                "sparse_raw": sparse_scores.get(doc_hash, 0.0),
                "sparse_normalized": sparse_norm,
                "dense_raw": dense_scores.get(doc_hash, 0.0),
                "dense_normalized": dense_norm,
                "final": final_score
            }
            
            fused_results.append((doc, final_score, score_breakdown))
        
        # Sort by final score and return top-k
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results[:top_k]
    
    def save(self) -> None:
        """Save both indices to disk."""
        Config.ensure_directories()
        self.sparse.save()
        self.dense.save()
    
    def load(self) -> None:
        """Load both indices from disk."""
        self.sparse.load()
        self.dense.load()
