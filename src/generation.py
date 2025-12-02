"""
Generation module for S.H.A.R.P.
Handles LLM generation using Google Gemini with citation-enforcing prompts.
"""

from typing import List, Tuple, Dict, Any, Optional

import google.generativeai as genai
from langchain_core.documents import Document

from .config import Config


class GeminiGenerator:
    """
    Generates answers using Google Gemini Pro with context from retrieval.
    Enforces citations to source documents in generated answers.
    """
    
    # Default prompt template enforcing citations
    DEFAULT_PROMPT_TEMPLATE = """You are an expert assistant that answers questions based on the provided context from scientific documents.

Context from retrieved documents:
{context}

Question: {question}

Instructions:
1. Answer the question using the information from the context above.
2. Cite your sources using the format: (Source: [Document Name], Page [X])
3. If the context contains relevant information, use it to provide a helpful answer.
4. If the context truly doesn't contain any relevant information, say "I cannot find sufficient information in the provided documents to answer this question."
5. Be precise and factual. Summarize key points from the documents.
6. If there are step-by-step procedures, list them clearly.

Answer:"""

    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = Config.LLM_MODEL
    ):
        """
        Initialize the Gemini generator.
        
        Args:
            api_key: Google API key (default: from Config)
            model_name: Name of the Gemini model to use
        """
        self.api_key = api_key or Config.GOOGLE_API_KEY
        self.model_name = model_name
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def format_context(
        self, 
        documents: List[Tuple[Document, float, Dict[str, float]]]
    ) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            documents: List of (Document, score, metadata) tuples
            
        Returns:
            Formatted context string with source information
        """
        context_parts = []
        
        for i, (doc, score, metadata) in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            
            context_part = f"""--- Document {i} ---
Source: {source}
Page: {page}
Content:
{doc.page_content}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def format_context_simple(
        self, 
        documents: List[Tuple[Document, float]]
    ) -> str:
        """
        Format retrieved documents (simple format without metadata dict).
        
        Args:
            documents: List of (Document, score) tuples
            
        Returns:
            Formatted context string with source information
        """
        context_parts = []
        
        for i, (doc, score) in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            
            context_part = f"""--- Document {i} ---
Source: {source}
Page: {page}
Content:
{doc.page_content}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate(
        self,
        question: str,
        documents: List[Tuple[Document, float, Dict[str, float]]],
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Generate an answer using the provided context.
        
        Args:
            question: The user's question
            documents: List of (Document, score, metadata) tuples
            prompt_template: Custom prompt template (optional)
            
        Returns:
            Generated answer string
        """
        if not documents:
            return "No relevant documents were found to answer your question."
        
        template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        context = self.format_context(documents)
        
        # Debug: Check if context is empty
        if not context or context.strip() == "":
            return "Error: Context is empty. Please check if documents were properly processed."
        
        prompt = template.format(context=context, question=question)
        
        # Debug: Print context length
        print(f"[DEBUG] Context length: {len(context)} chars, Documents: {len(documents)}")
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_simple(
        self,
        question: str,
        documents: List[Tuple[Document, float]],
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Generate an answer using simple document format.
        
        Args:
            question: The user's question
            documents: List of (Document, score) tuples
            prompt_template: Custom prompt template (optional)
            
        Returns:
            Generated answer string
        """
        if not documents:
            return "No relevant documents were found to answer your question."
        
        template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        context = self.format_context_simple(documents)
        
        prompt = template.format(context=context, question=question)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_with_debug(
        self,
        question: str,
        documents: List[Tuple[Document, float, Dict[str, float]]],
        prompt_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer with debug information.
        
        Args:
            question: The user's question
            documents: List of (Document, score, metadata) tuples
            prompt_template: Custom prompt template (optional)
            
        Returns:
            Dictionary containing answer, prompt, and document info
        """
        template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        context = self.format_context(documents)
        prompt = template.format(context=context, question=question)
        
        answer = self.generate(question, documents, prompt_template)
        
        # Extract document info for debug
        doc_info = []
        for i, (doc, score, metadata) in enumerate(documents, 1):
            doc_info.append({
                "position": i,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "score": score,
                "metadata": metadata,
                "content_preview": doc.page_content[:200] + "..."
            })
        
        return {
            "answer": answer,
            "prompt": prompt,
            "documents": doc_info,
            "num_documents": len(documents)
        }
