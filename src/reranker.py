"""
Reranker module for S.H.A.R.P.
Uses Cross-Encoder for semantic re-ranking of retrieved documents.
"""

from typing import List, Tuple, Union, Dict, Any, Optional

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from .config import Config


class CrossEncoderReranker:
    """
    Re-ranks documents using a Cross-Encoder model.
    Takes query-document pairs and scores their relevance.
    """
    
    def __init__(self, model_name: str = Config.RERANKER_MODEL):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model
        """
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents based on query relevance.
        
        Args:
            query: The search query
            documents: List of Document objects to re-rank
            top_n: Number of documents to return (default: all)
            
        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Combine documents with scores
        doc_scores = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convert numpy floats to Python floats
        doc_scores = [(doc, float(score)) for doc, score in doc_scores]
        
        if top_n is not None:
            return doc_scores[:top_n]
        
        return doc_scores
    
    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Tuple[Document, float, Dict[str, float]]],
        top_n: Optional[int] = None
    ) -> List[Tuple[Document, float, Dict[str, float]]]:
        """
        Re-rank documents while preserving retrieval metadata.
        
        Args:
            query: The search query
            documents: List of (Document, score, metadata) tuples from hybrid retriever
            top_n: Number of documents to return (default: all)
            
        Returns:
            List of (Document, rerank_score, metadata) tuples with added rerank info
        """
        if not documents:
            return []
        
        # Extract just the documents
        docs = [doc for doc, _, _ in documents]
        
        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in docs]
        
        # Get cross-encoder scores
        rerank_scores = self.model.predict(pairs)
        
        # Combine with original metadata
        results = []
        for i, (doc, orig_score, metadata) in enumerate(documents):
            updated_metadata = {
                **metadata,
                "rerank_score": float(rerank_scores[i]),
                "pre_rerank_position": i + 1
            }
            results.append((doc, float(rerank_scores[i]), updated_metadata))
        
        # Sort by rerank score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Add post-rerank position
        for i, (doc, score, metadata) in enumerate(results):
            metadata["post_rerank_position"] = i + 1
        
        if top_n is not None:
            return results[:top_n]
        
        return results
