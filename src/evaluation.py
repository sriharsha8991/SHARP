"""
Evaluation module for S.H.A.R.P.
Implements retrieval and generation quality metrics.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.metrics import ndcg_score
from langchain_core.documents import Document

from .config import Config


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.
    Supports NDCG@K, Recall@K, and Precision@K.
    """
    
    @staticmethod
    def calculate_recall_at_k(
        retrieved_docs: List[Document],
        relevant_doc_ids: List[str],
        k: int = 10
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved_docs: List of retrieved Document objects
            relevant_doc_ids: List of relevant document identifiers
            k: Cutoff for evaluation
            
        Returns:
            Recall score (0-1)
        """
        if not relevant_doc_ids:
            return 0.0
        
        retrieved_ids = set()
        for doc in retrieved_docs[:k]:
            # Use source + page as identifier
            doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}"
            retrieved_ids.add(doc_id)
        
        relevant_set = set(relevant_doc_ids)
        hits = len(retrieved_ids.intersection(relevant_set))
        
        return hits / len(relevant_set)
    
    @staticmethod
    def calculate_precision_at_k(
        retrieved_docs: List[Document],
        relevant_doc_ids: List[str],
        k: int = 10
    ) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved_docs: List of retrieved Document objects
            relevant_doc_ids: List of relevant document identifiers
            k: Cutoff for evaluation
            
        Returns:
            Precision score (0-1)
        """
        if not retrieved_docs or k == 0:
            return 0.0
        
        retrieved_ids = []
        for doc in retrieved_docs[:k]:
            doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}"
            retrieved_ids.append(doc_id)
        
        relevant_set = set(relevant_doc_ids)
        hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)
        
        return hits / min(k, len(retrieved_ids))
    
    @staticmethod
    def calculate_ndcg_at_k(
        retrieved_docs: List[Tuple[Document, float]],
        relevance_labels: Dict[str, int],
        k: int = 10
    ) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain).
        
        Args:
            retrieved_docs: List of (Document, score) tuples
            relevance_labels: Dict mapping doc_id to relevance score (0, 1, 2, ...)
            k: Cutoff for evaluation
            
        Returns:
            NDCG score (0-1)
        """
        if not retrieved_docs or not relevance_labels:
            return 0.0
        
        # Get relevance scores for retrieved documents
        y_true = []
        for doc, _ in retrieved_docs[:k]:
            doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}"
            rel_score = relevance_labels.get(doc_id, 0)
            y_true.append(rel_score)
        
        # Pad with zeros if needed
        while len(y_true) < k:
            y_true.append(0)
        
        # Create ideal ranking
        ideal_scores = sorted(relevance_labels.values(), reverse=True)[:k]
        while len(ideal_scores) < k:
            ideal_scores.append(0)
        
        # Calculate NDCG using sklearn
        try:
            score = ndcg_score([ideal_scores], [y_true])
            return float(score)
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_mrr(
        retrieved_docs: List[Document],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_docs: List of retrieved Document objects
            relevant_doc_ids: List of relevant document identifiers
            
        Returns:
            MRR score (0-1)
        """
        if not relevant_doc_ids:
            return 0.0
        
        relevant_set = set(relevant_doc_ids)
        
        for i, doc in enumerate(retrieved_docs, 1):
            doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}"
            if doc_id in relevant_set:
                return 1.0 / i
        
        return 0.0


class GenerationEvaluator:
    """
    Evaluates generation quality using semantic similarity metrics.
    """
    
    def __init__(self):
        """Initialize the generation evaluator."""
        self._bert_score = None
    
    @property
    def bert_scorer(self):
        """Lazy load BERTScore to avoid import overhead."""
        if self._bert_score is None:
            from bert_score import BERTScorer
            self._bert_score = BERTScorer(lang="en", rescale_with_baseline=True)
        return self._bert_score
    
    def calculate_bert_score(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate BERTScore for generated answers.
        
        Args:
            predictions: List of generated answers
            references: List of reference answers
            
        Returns:
            Dict with precision, recall, and F1 scores
        """
        P, R, F1 = self.bert_scorer.score(predictions, references)
        
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean())
        }
    
    @staticmethod
    def calculate_answer_relevance(
        question: str,
        answer: str,
        generator: 'GeminiGenerator'
    ) -> float:
        """
        Use LLM-as-a-judge to score answer relevance.
        
        Args:
            question: The original question
            answer: The generated answer
            generator: GeminiGenerator instance for judging
            
        Returns:
            Relevance score (0-10)
        """
        judge_prompt = f"""Rate the relevance and quality of the following answer on a scale of 0-10.

Question: {question}

Answer: {answer}

Scoring criteria:
- 0-2: Completely irrelevant or wrong
- 3-4: Partially relevant but missing key information
- 5-6: Relevant but could be more complete
- 7-8: Good answer with most key points
- 9-10: Excellent, comprehensive answer

Respond with ONLY a number from 0-10."""

        try:
            response = generator.model.generate_content(judge_prompt)
            score_text = response.text.strip()
            score = float(score_text)
            return min(max(score, 0), 10)  # Clamp to 0-10
        except Exception:
            return 5.0  # Default middle score on error
    
    @staticmethod
    def calculate_faithfulness(
        answer: str,
        context: str,
        generator: 'GeminiGenerator'
    ) -> float:
        """
        Use LLM-as-a-judge to score faithfulness to context.
        
        Args:
            answer: The generated answer
            context: The source context
            generator: GeminiGenerator instance for judging
            
        Returns:
            Faithfulness score (0-10)
        """
        judge_prompt = f"""Rate how faithful the answer is to the provided context on a scale of 0-10.

Context:
{context[:2000]}  # Truncate for token limits

Answer: {answer}

Scoring criteria:
- 0-2: Contains significant hallucinations or fabrications
- 3-4: Some claims not supported by context
- 5-6: Mostly faithful with minor unsupported claims
- 7-8: Faithful with proper citations
- 9-10: Completely faithful, all claims supported by context

Respond with ONLY a number from 0-10."""

        try:
            response = generator.model.generate_content(judge_prompt)
            score_text = response.text.strip()
            score = float(score_text)
            return min(max(score, 0), 10)
        except Exception:
            return 5.0


class EvaluationPipeline:
    """
    Complete evaluation pipeline for the SHARP system.
    Runs retrieval and generation evaluations across multiple modes.
    """
    
    def __init__(
        self,
        sparse_retriever=None,
        dense_retriever=None,
        hybrid_retriever=None,
        reranker=None,
        generator=None
    ):
        """
        Initialize the evaluation pipeline.
        
        Args:
            sparse_retriever: SparseRetriever instance
            dense_retriever: DenseRetriever instance
            hybrid_retriever: HybridRetriever instance
            reranker: CrossEncoderReranker instance
            generator: GeminiGenerator instance
        """
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.hybrid = hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        
        self.retrieval_eval = RetrievalEvaluator()
        self.generation_eval = GenerationEvaluator()
    
    def load_ground_truth(self, json_path: Path) -> List[Dict[str, Any]]:
        """
        Load ground truth dataset from JSON.
        
        Expected format:
        [
            {
                "question": "...",
                "answer": "...",  # Reference answer
                "relevant_docs": ["source_page", ...]  # Relevant doc IDs
            },
            ...
        ]
        """
        with open(json_path, "r") as f:
            return json.load(f)
    
    def evaluate_retrieval_mode(
        self,
        mode: str,
        questions: List[str],
        relevant_docs_list: List[List[str]],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate a single retrieval mode.
        
        Args:
            mode: "sparse", "dense", or "hybrid"
            questions: List of questions
            relevant_docs_list: List of relevant doc ID lists
            k: Cutoff for metrics
            
        Returns:
            Dict with averaged metrics
        """
        retriever = {
            "sparse": self.sparse,
            "dense": self.dense,
            "hybrid": self.hybrid
        }.get(mode)
        
        if retriever is None:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        
        recalls = []
        precisions = []
        mrrs = []
        
        for question, relevant_docs in zip(questions, relevant_docs_list):
            if mode == "hybrid":
                results = retriever.retrieve(question, top_k=k)
                docs = [doc for doc, _, _ in results]
            else:
                results = retriever.retrieve(question, top_k=k)
                docs = [doc for doc, _ in results]
            
            recalls.append(
                self.retrieval_eval.calculate_recall_at_k(docs, relevant_docs, k)
            )
            precisions.append(
                self.retrieval_eval.calculate_precision_at_k(docs, relevant_docs, k)
            )
            mrrs.append(
                self.retrieval_eval.calculate_mrr(docs, relevant_docs)
            )
        
        return {
            f"recall@{k}": np.mean(recalls),
            f"precision@{k}": np.mean(precisions),
            "mrr": np.mean(mrrs)
        }
    
    def run_full_evaluation(
        self,
        ground_truth_path: Path,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Run complete evaluation across all modes.
        
        Args:
            ground_truth_path: Path to ground truth JSON
            k: Cutoff for metrics
            
        Returns:
            Comprehensive evaluation results
        """
        data = self.load_ground_truth(ground_truth_path)
        
        questions = [item["question"] for item in data]
        relevant_docs_list = [item.get("relevant_docs", []) for item in data]
        reference_answers = [item.get("answer", "") for item in data]
        
        results = {
            "retrieval": {},
            "generation": {}
        }
        
        # Evaluate all retrieval modes
        for mode in ["sparse", "dense", "hybrid"]:
            try:
                results["retrieval"][mode] = self.evaluate_retrieval_mode(
                    mode, questions, relevant_docs_list, k
                )
            except Exception as e:
                results["retrieval"][mode] = {"error": str(e)}
        
        # Evaluate generation (using hybrid + reranker by default)
        if self.hybrid and self.reranker and self.generator:
            generated_answers = []
            
            for question in questions:
                # Get hybrid results
                hybrid_results = self.hybrid.retrieve(question, top_k=k)
                
                # Rerank
                reranked = self.reranker.rerank_with_metadata(
                    question, hybrid_results, top_n=5
                )
                
                # Generate
                answer = self.generator.generate(question, reranked)
                generated_answers.append(answer)
            
            # Calculate BERTScore
            if reference_answers and all(reference_answers):
                results["generation"]["bert_score"] = \
                    self.generation_eval.calculate_bert_score(
                        generated_answers, reference_answers
                    )
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save evaluation results to JSON."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
