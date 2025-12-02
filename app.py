"""
S.H.A.R.P. - Scientific Hybrid Augmented Retrieval Pipeline
Streamlit Application Entry Point
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from src.config import Config
from src.ingestion import DocumentProcessor
from src.retrieval import SparseRetriever, DenseRetriever, HybridRetriever
from src.reranker import CrossEncoderReranker
from src.generation import GeminiGenerator


# Page configuration
st.set_page_config(
    page_title="S.H.A.R.P.",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "documents" not in st.session_state:
        st.session_state.documents = []
    
    if "sparse_retriever" not in st.session_state:
        st.session_state.sparse_retriever = None
    
    if "dense_retriever" not in st.session_state:
        st.session_state.dense_retriever = None
    
    if "hybrid_retriever" not in st.session_state:
        st.session_state.hybrid_retriever = None
    
    if "reranker" not in st.session_state:
        st.session_state.reranker = None
    
    if "generator" not in st.session_state:
        st.session_state.generator = None
    
    if "indexed" not in st.session_state:
        st.session_state.indexed = False


def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files and build indices."""
    with st.spinner("Processing documents..."):
        processor = DocumentProcessor()
        all_chunks = []
        
        for uploaded_file in uploaded_files:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = Path(tmp.name)
            
            # Process the PDF
            chunks = processor.process_uploaded_file(tmp_path)
            all_chunks.extend(chunks)
            
            # Clean up temp file
            os.unlink(tmp_path)
        
        st.session_state.documents = all_chunks
        st.success(f"Processed {len(uploaded_files)} files into {len(all_chunks)} chunks")
    
    # Build indices
    with st.spinner("Building search indices..."):
        # Initialize retrievers
        st.session_state.sparse_retriever = SparseRetriever()
        st.session_state.dense_retriever = DenseRetriever()
        st.session_state.hybrid_retriever = HybridRetriever(
            st.session_state.sparse_retriever,
            st.session_state.dense_retriever
        )
        
        # Build indices
        st.session_state.hybrid_retriever.index(all_chunks)
        
        # Initialize reranker
        st.session_state.reranker = CrossEncoderReranker()
        
        st.session_state.indexed = True
        st.success("Indices built successfully!")


def initialize_generator():
    """Initialize the Gemini generator."""
    if st.session_state.generator is None:
        try:
            st.session_state.generator = GeminiGenerator()
        except Exception as e:
            st.error(f"Failed to initialize generator: {e}")
            st.info("Please set your GOOGLE_API_KEY in the .env file")
            return False
    return True


def perform_retrieval(query: str, mode: str, alpha: float, top_k: int):
    """Perform retrieval based on selected mode."""
    if mode == "Sparse (BM25)":
        results = st.session_state.sparse_retriever.retrieve(query, top_k)
        # Convert to hybrid format for consistency
        return [(doc, score, {"sparse_raw": score, "final": score}) for doc, score in results]
    
    elif mode == "Dense (Embeddings)":
        results = st.session_state.dense_retriever.retrieve(query, top_k)
        return [(doc, score, {"dense_raw": score, "final": score}) for doc, score in results]
    
    else:  # Hybrid
        return st.session_state.hybrid_retriever.retrieve(query, alpha, top_k)


def display_debug_info(retrieved_docs, reranked_docs, mode: str):
    """Display debug information about retrieved documents."""
    with st.expander("🔍 Debug: Retrieved Context", expanded=False):
        st.markdown("### Retrieval Results")
        st.markdown(f"**Mode:** {mode}")
        
        # Show retrieved documents before reranking
        st.markdown("#### Before Re-ranking")
        for i, (doc, score, metadata) in enumerate(retrieved_docs[:5], 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            
            st.markdown(f"""
            **{i}. {source} (Page {page})**
            - Final Score: `{metadata.get('final', score):.4f}`
            """)
            
            if "sparse_normalized" in metadata:
                st.markdown(f"""
                - Sparse (normalized): `{metadata.get('sparse_normalized', 0):.4f}`
                - Dense (normalized): `{metadata.get('dense_normalized', 0):.4f}`
                """)
            
            with st.container():
                st.text(doc.page_content[:300] + "...")
            
            st.divider()
        
        # Show reranked documents
        st.markdown("#### After Re-ranking (Cross-Encoder)")
        for i, (doc, score, metadata) in enumerate(reranked_docs[:5], 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            
            pre_pos = metadata.get("pre_rerank_position", "?")
            post_pos = metadata.get("post_rerank_position", i)
            
            st.markdown(f"""
            **{i}. {source} (Page {page})**
            - Rerank Score: `{score:.4f}`
            - Position Change: #{pre_pos} → #{post_pos}
            """)
            
            with st.container():
                st.text(doc.page_content[:200] + "...")
            
            st.divider()


def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Header
    st.title("🔬 S.H.A.R.P.")
    st.markdown("**Scientific Hybrid Augmented Retrieval Pipeline**")
    st.markdown("*Extract actionable insights from scientific PDFs with hybrid retrieval*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File uploader
        st.subheader("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload scientific PDFs to search through"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                process_uploaded_files(uploaded_files)
        
        st.divider()
        
        # Retrieval mode selector
        st.subheader("🔎 Retrieval Mode")
        retrieval_mode = st.selectbox(
            "Select retrieval method",
            options=["Hybrid (Recommended)", "Sparse (BM25)", "Dense (Embeddings)"],
            help="Hybrid combines BM25 and semantic search for best results"
        )
        
        # Alpha slider for hybrid mode
        if retrieval_mode == "Hybrid (Recommended)":
            alpha = st.slider(
                "Alpha (Dense Weight)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="0 = Pure BM25, 1 = Pure Dense, 0.5 = Equal weight"
            )
        else:
            alpha = 0.5
        
        # Top-K slider
        top_k = st.slider(
            "Top-K Results",
            min_value=3,
            max_value=20,
            value=10,
            help="Number of documents to retrieve"
        )
        
        st.divider()
        
        # Status indicators
        st.subheader("📊 Status")
        if st.session_state.indexed:
            st.success(f"✅ {len(st.session_state.documents)} chunks indexed")
        else:
            st.warning("⚠️ No documents indexed")
        
        if st.session_state.generator:
            st.success("✅ Generator ready")
        else:
            st.info("ℹ️ Generator will initialize on first query")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.divider()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show debug info for assistant messages
            if message["role"] == "assistant" and "debug" in message:
                display_debug_info(
                    message["debug"]["retrieved"],
                    message["debug"]["reranked"],
                    message["debug"]["mode"]
                )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if documents are indexed
        if not st.session_state.indexed:
            st.error("Please upload and process documents first!")
            return
        
        # Initialize generator if needed
        if not initialize_generator():
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                # Perform retrieval
                retrieved_docs = perform_retrieval(
                    prompt, retrieval_mode, alpha, top_k
                )
                
                # Rerank
                reranked_docs = st.session_state.reranker.rerank_with_metadata(
                    prompt, retrieved_docs, top_n=5
                )
                
                # Generate answer
                response = st.session_state.generator.generate(
                    prompt, reranked_docs
                )
                
                st.markdown(response)
                
                # Store debug info
                debug_info = {
                    "retrieved": retrieved_docs,
                    "reranked": reranked_docs,
                    "mode": retrieval_mode
                }
                
                # Display debug
                display_debug_info(retrieved_docs, reranked_docs, retrieval_mode)
        
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "debug": debug_info
        })


if __name__ == "__main__":
    # Ensure directories exist
    Config.ensure_directories()
    
    # Run the app
    main()
