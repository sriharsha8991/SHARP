

# S.H.A.R.P. (Scientific Hybrid Augmented Retrieval Pipeline)

## 📌 Project Overview
**S.H.A.R.P.** is a domain-adaptive Hybrid RAG system designed to extract actionable steps from scientific PDFs.
It contrasts **Sparse Retrieval (BM25)**, **Dense Retrieval (Embeddings)**, and **Hybrid Retrieval (Weighted Fusion)**, followed by a **Cross-Encoder Re-ranking** step and **LLM Generation** (Google Gemini).

**Objective:** Minimize hallucinations and improve factual consistency in procedural QA.

## 🏗 System Architecture
The pipeline follows this strictly defined flow:
1.  **Ingestion:** Load PDFs $\rightarrow$ Clean Text $\rightarrow$ Chunk (Recursive Split).
2.  **Indexing:**
    *   **Sparse:** Tokenize and build BM25 Index.
    *   **Dense:** Generate embeddings (`all-mpnet-base-v2`) and store in FAISS/Qdrant.
3.  **Retrieval:**
    *   Fetch Top-K from BM25.
    *   Fetch Top-K from Vector DB.
    *   **Fusion:** Normalize scores $\rightarrow$ Weighted Sum (Alpha parameter).
4.  **Re-Ranking:** Pass Top-N fused results into a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
5.  **Generation:** Construct prompt with re-ranked context $\rightarrow$ Query Gemini Pro API.

---

## 📂 Directory Structure (Context for AI)
*Use this structure when generating files.*

```text
SHARP_Project/
│
├── data/                     # Folder for source PDF files
├── indices/                  # Folder to save local FAISS/BM25 indices
├── src/
│   ├── __init__.py
│   ├── config.py             # Global variables (API Keys, Chunk Sizes, Model Names)
│   ├── ingestion.py          # PDF loading, cleaning, and chunking logic
│   ├── retrieval.py          # The Core Engine: Sparse, Dense, and Hybrid classes
│   ├── reranker.py           # Cross-Encoder logic
│   ├── generation.py         # Google Gemini Client & Prompt Templates
│   └── evaluation.py         # RAGAS metrics, NDCG, BERTScore implementation
│
├── app.py                    # Streamlit UI entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🛠 Technical Specifications for Code Generation

### 1. Configuration (`src/config.py`)
*   **LLM:** `gemini-pro` (via `google-generativeai`).
*   **Embedding Model:** `sentence-transformers/all-mpnet-base-v2`.
*   **Reranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
*   **Chunking:** Size=500, Overlap=50.
*   **Top-K Retrieval:** 10 docs per method.

### 2. Ingestion (`src/ingestion.py`)
*   Use `PyPDFLoader` (LangChain) or `pypdf`.
*   Clean text (remove newlines/citations like `[1]`).
*   **Critical:** Metadata for every chunk must include `{"source": filename, "page": page_num}`.

### 3. Retrieval Logic (`src/retrieval.py`)
*   **Class `SparseRetriever`:** Use `rank_bm25`. Tokenize text before indexing.
*   **Class `DenseRetriever`:** Use `FAISS` (CPU) or `Qdrant`.
*   **Class `HybridRetriever`:**
    *   **Method:** `retrieve(query, alpha=0.5)`
    *   **Normalization:** BM25 scores range from 0 to 50+, Cosine is 0 to 1. Implement **Min-Max Normalization** on BM25 scores before combining.
    *   **Formula:** $Score_{final} = (\alpha \times Score_{dense}) + ((1-\alpha) \times Score_{sparse})$

### 4. Re-ranking (`src/reranker.py`)
*   Input: Query + List of Documents.
*   Logic: Use `CrossEncoder` from `sentence_transformers`.
*   Output: Sorted list of documents based on relevance score.

### 5. Generation (`src/generation.py`)
*   **Prompt Template:**
    ```text
    Context: {context}
    Question: {question}
    Instruction: Answer based ONLY on the context. Cite the source (Document Name, Page Number) for every claim.
    ```

### 6. User Interface (`app.py`)
*   Use `Streamlit`.
*   **Sidebar:** File Uploader (re-runs ingestion on upload), "Select Retrieval Mode" (Sparse/Dense/Hybrid).
*   **Main:** Chat interface.
*   **Debug Expander:** Show "Retrieved Context" with scores to demonstrate the hybrid merging.

---

## 📦 Requirements
*Create a `requirements.txt` with these:*
```text
langchain
langchain-community
langchain-google-genai
sentence-transformers
faiss-cpu
rank_bm25
pypdf
streamlit
python-dotenv
google-generativeai
scikit-learn
bert-score
evaluate
```

---

## 🚀 How to Build (For Copilot/AI)
1.  **Setup:** Create `src/config.py` and load `.env` for `GOOGLE_API_KEY`.
2.  **Data:** Implement `src/ingestion.py` to process PDFs in the `data/` folder.
3.  **Indices:** Implement `src/retrieval.py` to save/load FAISS and BM25 pickles.
4.  **Pipeline:** Connect Retrieval $\rightarrow$ Reranker $\rightarrow$ Generator in `app.py`.
5.  **Run:** Execute `streamlit run app.py`.

---

## 🧪 Evaluation Plan
The system includes an `evaluation.py` script that:
1.  Takes a ground truth dataset (JSON).
2.  Runs retrieval for all 3 modes.
3.  Calculates **NDCG@10** and **Recall@10**.
4.  Uses LLM-as-a-judge to score the final answer.