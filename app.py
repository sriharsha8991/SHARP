"""
S.H.A.R.P. - Scientific Hybrid Augmented Retrieval Pipeline
Streamlit Application Entry Point
"""

import os
import tempfile
from pathlib import Path
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import Config
from src.ingestion import DocumentProcessor
from src.retrieval import SparseRetriever, DenseRetriever, HybridRetriever
from src.reranker import CrossEncoderReranker
from src.generation import GeminiGenerator
from src.evaluation import RetrievalEvaluator, GenerationEvaluator


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
    
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = []
    
    if "eval_queries" not in st.session_state:
        st.session_state.eval_queries = []


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
    
    # Main content with tabs
    tab1, tab2 = st.tabs(["💬 Chat Interface", "📊 Evaluation & Metrics"])
    
    with tab1:
        render_chat_tab(retrieval_mode, alpha, top_k)
    
    with tab2:
        render_evaluation_tab(retrieval_mode, alpha, top_k)


def render_chat_tab(retrieval_mode: str, alpha: float, top_k: int):
    """Render the chat interface tab."""
    st.markdown("### Ask questions about your documents")
    
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


def render_evaluation_tab(retrieval_mode: str, alpha: float, top_k: int):
    """Render the evaluation and metrics tab."""
    st.markdown("### 📈 System Evaluation & Performance Metrics")
    
    if not st.session_state.indexed:
        st.warning("⚠️ Please upload and process documents first to run evaluations")
        return
    
    # Section 1: Quick Comparison
    st.subheader("🔍 Compare Retrieval Methods")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_query = st.text_input(
            "Enter a test query",
            placeholder="What are the main findings?",
            help="Test how different retrieval methods perform on the same query"
        )
    
    with col2:
        compare_top_k = st.number_input("Top-K", min_value=3, max_value=20, value=5)
    
    if test_query and st.button("🔎 Run Comparison", type="primary"):
        run_retrieval_comparison(test_query, compare_top_k, alpha)
    
    st.divider()
    
    # Section 2: Batch Evaluation
    st.subheader("📋 Batch Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Option 1: Upload Evaluation Dataset**")
        eval_file = st.file_uploader(
            "Upload JSON evaluation file",
            type=["json"],
            help="Format: {\"queries\": [...], \"relevant_docs\": [[...]], \"reference_answers\": [...]}"
        )
        
        if eval_file:
            try:
                eval_data = json.load(eval_file)
                st.success(f"✅ Loaded {len(eval_data.get('queries', []))} evaluation queries")
                
                if st.button("Run Batch Evaluation"):
                    run_batch_evaluation(eval_data, alpha)
            except Exception as e:
                st.error(f"Error loading evaluation file: {e}")
    
    with col2:
        st.markdown("**Option 2: Manual Test Queries**")
        
        num_queries = st.number_input("Number of test queries", min_value=1, max_value=10, value=3)
        
        manual_queries = []
        for i in range(num_queries):
            query = st.text_input(f"Query {i+1}", key=f"manual_query_{i}")
            if query:
                manual_queries.append(query)
        
        if manual_queries and st.button("Run Manual Evaluation"):
            run_manual_evaluation(manual_queries, alpha)
    
    st.divider()
    
    # Section 3: Results Visualization
    if st.session_state.eval_results:
        st.subheader("📊 Evaluation Results")
        display_evaluation_results()
    
    st.divider()
    
    # Section 4: System Statistics
    st.subheader("📈 System Statistics")
    display_system_statistics()


def run_retrieval_comparison(query: str, top_k: int, alpha: float):
    """Compare all three retrieval methods on a single query."""
    with st.spinner("Running comparison across all retrieval methods..."):
        results = {}
        
        # Sparse Retrieval
        sparse_results = st.session_state.sparse_retriever.retrieve(query, top_k)
        results["Sparse (BM25)"] = sparse_results
        
        # Dense Retrieval
        dense_results = st.session_state.dense_retriever.retrieve(query, top_k)
        results["Dense (Embeddings)"] = dense_results
        
        # Hybrid Retrieval
        hybrid_results = st.session_state.hybrid_retriever.retrieve(query, alpha, top_k)
        results["Hybrid"] = [(doc, score) for doc, score, _ in hybrid_results]
        
        # Display comparison
        display_retrieval_comparison(query, results, top_k)


def display_retrieval_comparison(query: str, results: dict, top_k: int):
    """Display side-by-side comparison of retrieval methods."""
    st.markdown(f"**Query:** _{query}_")
    
    # Create columns for each method
    cols = st.columns(3)
    
    for idx, (method_name, docs) in enumerate(results.items()):
        with cols[idx]:
            st.markdown(f"### {method_name}")
            
            for i, (doc, score) in enumerate(docs[:top_k], 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                
                with st.expander(f"#{i} - {source} (p.{page}) | Score: {score:.3f}"):
                    st.text(doc.page_content[:300] + "...")
    
    # Score distribution comparison
    st.markdown("### Score Distribution Comparison")
    
    fig = go.Figure()
    
    for method_name, docs in results.items():
        scores = [score for _, score in docs[:top_k]]
        fig.add_trace(go.Bar(
            name=method_name,
            x=[f"Doc {i+1}" for i in range(len(scores))],
            y=scores
        ))
    
    fig.update_layout(
        title="Retrieval Scores by Method",
        xaxis_title="Document Rank",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def run_batch_evaluation(eval_data: dict, alpha: float):
    """Run batch evaluation on provided dataset."""
    queries = eval_data.get("queries", [])
    relevant_docs = eval_data.get("relevant_docs", [])
    reference_answers = eval_data.get("reference_answers", [])
    
    evaluator = RetrievalEvaluator()
    gen_evaluator = GenerationEvaluator()
    
    results = {
        "Sparse": {"ndcg": [], "recall": [], "precision": []},
        "Dense": {"ndcg": [], "recall": [], "precision": []},
        "Hybrid": {"ndcg": [], "recall": [], "precision": []}
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, query in enumerate(queries):
        status_text.text(f"Evaluating query {i+1}/{len(queries)}: {query[:50]}...")
        
        # Get relevant docs for this query
        relevant = relevant_docs[i] if i < len(relevant_docs) else []
        
        # Evaluate each method
        for method_name, retriever_type in [("Sparse", "sparse"), ("Dense", "dense"), ("Hybrid", "hybrid")]:
            if retriever_type == "sparse":
                retrieved = st.session_state.sparse_retriever.retrieve(query, 10)
            elif retriever_type == "dense":
                retrieved = st.session_state.dense_retriever.retrieve(query, 10)
            else:
                retrieved = st.session_state.hybrid_retriever.retrieve(query, alpha, 10)
                retrieved = [(doc, score) for doc, score, _ in retrieved]
            
            # Calculate metrics
            docs_only = [doc for doc, _ in retrieved]
            
            if relevant:
                recall = evaluator.calculate_recall_at_k(docs_only, relevant, 10)
                precision = evaluator.calculate_precision_at_k(docs_only, relevant, 10)
                results[method_name]["recall"].append(recall)
                results[method_name]["precision"].append(precision)
        
        progress_bar.progress((i + 1) / len(queries))
    
    status_text.text("✅ Evaluation complete!")
    
    # Store results
    st.session_state.eval_results.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "batch",
        "num_queries": len(queries),
        "results": results
    })
    
    st.success(f"Evaluated {len(queries)} queries across 3 retrieval methods")


def run_manual_evaluation(queries: list, alpha: float):
    """Run evaluation on manually entered queries."""
    results = {
        "queries": queries,
        "sparse_scores": [],
        "dense_scores": [],
        "hybrid_scores": []
    }
    
    for query in queries:
        # Sparse
        sparse_results = st.session_state.sparse_retriever.retrieve(query, 5)
        avg_sparse = sum(score for _, score in sparse_results) / len(sparse_results) if sparse_results else 0
        results["sparse_scores"].append(avg_sparse)
        
        # Dense
        dense_results = st.session_state.dense_retriever.retrieve(query, 5)
        avg_dense = sum(score for _, score in dense_results) / len(dense_results) if dense_results else 0
        results["dense_scores"].append(avg_dense)
        
        # Hybrid
        hybrid_results = st.session_state.hybrid_retriever.retrieve(query, alpha, 5)
        avg_hybrid = sum(score for _, score, _ in hybrid_results) / len(hybrid_results) if hybrid_results else 0
        results["hybrid_scores"].append(avg_hybrid)
    
    # Store results
    st.session_state.eval_results.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "manual",
        "results": results
    })
    
    st.success(f"✅ Evaluated {len(queries)} manual queries")


def display_evaluation_results():
    """Display stored evaluation results."""
    if not st.session_state.eval_results:
        st.info("No evaluation results yet. Run an evaluation above.")
        return
    
    # Display latest results
    latest = st.session_state.eval_results[-1]
    
    st.markdown(f"**Latest Evaluation:** {latest['timestamp']} ({latest['type']})")
    
    if latest['type'] == 'batch':
        results = latest['results']
        
        # Create metrics DataFrame
        metrics_data = []
        for method in ["Sparse", "Dense", "Hybrid"]:
            if results[method]["recall"]:
                metrics_data.append({
                    "Method": method,
                    "Avg Recall@10": f"{sum(results[method]['recall'])/len(results[method]['recall']):.4f}",
                    "Avg Precision@10": f"{sum(results[method]['precision'])/len(results[method]['precision']):.4f}",
                })
        
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        
        for method in ["Sparse", "Dense", "Hybrid"]:
            if results[method]["recall"]:
                avg_recall = sum(results[method]["recall"]) / len(results[method]["recall"])
                avg_precision = sum(results[method]["precision"]) / len(results[method]["precision"])
                
                fig.add_trace(go.Bar(
                    name=method,
                    x=["Recall@10", "Precision@10"],
                    y=[avg_recall, avg_precision]
                ))
        
        fig.update_layout(
            title="Retrieval Performance Comparison",
            yaxis_title="Score",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif latest['type'] == 'manual':
        results = latest['results']
        
        # Create DataFrame
        df = pd.DataFrame({
            "Query": results["queries"],
            "Sparse Score": results["sparse_scores"],
            "Dense Score": results["dense_scores"],
            "Hybrid Score": results["hybrid_scores"]
        })
        
        st.dataframe(df, use_container_width=True)


def display_system_statistics():
    """Display system statistics and information."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Documents",
            len(st.session_state.documents) if st.session_state.indexed else 0
        )
    
    with col2:
        st.metric(
            "Chat Interactions",
            len([m for m in st.session_state.messages if m["role"] == "user"])
        )
    
    with col3:
        st.metric(
            "Evaluations Run",
            len(st.session_state.eval_results)
        )
    
    # Model information
    with st.expander("🔧 Model Configuration"):
        st.markdown(f"""
        **LLM Model:** `{Config.LLM_MODEL}`  
        **Embedding Model:** `{Config.EMBEDDING_MODEL}`  
        **Reranker Model:** `{Config.RERANKER_MODEL}`  
        **Chunk Size:** {Config.CHUNK_SIZE} chars  
        **Chunk Overlap:** {Config.CHUNK_OVERLAP} chars  
        **Default Top-K:** {Config.TOP_K}  
        **Default Alpha:** {Config.DEFAULT_ALPHA}
        """)
    
    # Export functionality
    if st.session_state.eval_results:
        st.markdown("### 💾 Export Results")
        
        if st.button("Download Evaluation Results as JSON"):
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "evaluation_results": st.session_state.eval_results,
                "system_config": {
                    "llm_model": Config.LLM_MODEL,
                    "embedding_model": Config.EMBEDDING_MODEL,
                    "reranker_model": Config.RERANKER_MODEL
                }
            }
            
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"sharp_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    # Ensure directories exist
    Config.ensure_directories()
    
    # Run the app
    main()
