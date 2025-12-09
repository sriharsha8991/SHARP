

# S.H.A.R.P. (Scientific Hybrid Augmented Retrieval Pipeline)

## 📌 Project Overview
**S.H.A.R.P.** is a production-ready Hybrid RAG system designed to extract actionable insights from scientific PDFs with minimal hallucinations.

**Key Features:**
- Three retrieval strategies: **Sparse (BM25)**, **Dense (FAISS)**, and **Hybrid (Weighted Fusion)**
- **Cross-Encoder Re-ranking** with `ms-marco-MiniLM-L-6-v2`
- **Citation-enforced LLM Generation** using Google Gemini Flash Lite
- **Interactive Streamlit UI** with debug visualization
- **Comprehensive evaluation** metrics (NDCG, Recall, BERTScore)

**Objective:** Minimize hallucinations and improve factual consistency in scientific question-answering.

## 🏗 System Architecture
The pipeline implements a complete RAG flow with the following stages:

1.  **Ingestion Pipeline:**
    - Load PDFs using `PyPDFLoader` from LangChain
    - Clean text (remove citation markers `[1]`, normalize whitespace)
    - Chunk documents using `RecursiveCharacterTextSplitter`
    - Preserve metadata: `source` (filename), `page` (page number)

2.  **Indexing (Dual Strategy):**
    - **Sparse Index:** Tokenize and build BM25Okapi index with simple whitespace tokenization
    - **Dense Index:** Generate 768-dim embeddings using `all-mpnet-base-v2`, store in FAISS IndexFlatIP with L2 normalization

3.  **Retrieval (Three Modes):**
    - **Sparse:** BM25-based keyword matching (Top-K documents)
    - **Dense:** Cosine similarity search in FAISS vector store (Top-K documents)
    - **Hybrid:** Min-max normalization → Weighted fusion: `score = (α × dense) + ((1-α) × sparse)`

4.  **Re-Ranking:**
    - Pass Top-N fused results to Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
    - Sort by semantic relevance scores
    - Preserve metadata including pre/post rerank positions

5.  **Generation:**
    - Construct citation-enforcing prompt with re-ranked context
    - Query Google Gemini Flash Lite API
    - Format: `(Source: [Document Name], Page [X])`

---

## 📂 Directory Structure

```text
SHARP/
│
├── data/                     # Source PDF files (upload via UI or place manually)
├── indices/                  # Saved FAISS/BM25 indices (auto-generated)
│
├── src/                      # Core pipeline modules
│   ├── __init__.py          # Package exports
│   ├── config.py            # Configuration (API keys, model names, hyperparameters)
│   ├── ingestion.py         # DocumentProcessor: PDF loading, cleaning, chunking
│   ├── retrieval.py         # SparseRetriever, DenseRetriever, HybridRetriever
│   ├── reranker.py          # CrossEncoderReranker with metadata preservation
│   ├── generation.py        # GeminiGenerator with citation-enforcing prompts
│   └── evaluation.py        # Metrics: NDCG@K, Recall@K, BERTScore
│
├── app.py                   # Streamlit UI with chat interface and debug visualization
├── requirements.txt         # Python dependencies (16 packages)
├── .env.example            # Template for environment variables
├── .gitignore              # Git exclusions (venv/, .env, indices/, etc.)
└── README.md               # This file
```

---

## 🛠 Technical Specifications

### Core Configuration (`src/config.py`)
```python
# LLM Configuration
LLM_MODEL = "models/gemini-flash-lite-latest"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Embedding & Reranking Models
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 768-dim embeddings
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking Parameters
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks

# Retrieval Parameters
TOP_K = 10                # Documents to retrieve per method
DEFAULT_ALPHA = 0.5       # Hybrid fusion weight (0=sparse only, 1=dense only)
```

### Ingestion Pipeline (`src/ingestion.py`)
**Class: `DocumentProcessor`**
- **PDF Loading:** `PyPDFLoader` from `langchain-community`
- **Text Cleaning:** 
  - Removes citation markers: `[1]`, `[2]`, etc.
  - Normalizes whitespace and newlines
- **Chunking Strategy:** `RecursiveCharacterTextSplitter`
  - Separators: `["\n\n", "\n", ". ", " ", ""]`
- **Metadata Preservation:** Every chunk includes:
  ```python
  {"source": "filename.pdf", "page": 5, "chunk_index": 12}
  ```

### Retrieval Engine (`src/retrieval.py`)

**1. SparseRetriever (BM25)**
```python
class SparseRetriever:
    - Algorithm: BM25Okapi from rank_bm25
    - Tokenization: Lowercase + whitespace split
    - Returns: List[Tuple[Document, float]] sorted by BM25 score
```

**2. DenseRetriever (FAISS)**
```python
class DenseRetriever:
    - Encoder: SentenceTransformer('all-mpnet-base-v2')
    - Index: FAISS IndexFlatIP (cosine similarity after L2 normalization)
    - Embedding Dim: 768
    - Returns: List[Tuple[Document, float]] sorted by cosine similarity
```

**3. HybridRetriever (Fusion)**
```python
class HybridRetriever:
    - Method: Min-Max Normalization + Weighted Sum
    - Formula: final_score = (α × dense_norm) + ((1-α) × sparse_norm)
    - Handles score range mismatch (BM25: 0-50+, Cosine: 0-1)
    - Returns: Merged and sorted results
```

### Re-ranking (`src/reranker.py`)
**Class: `CrossEncoderReranker`**
```python
- Model: CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
- Input: Query + List[Document]
- Process: Scores each (query, doc) pair
- Sorting: Descending by relevance score
- Metadata: Adds pre_rerank_position, post_rerank_position
```

### Generation (`src/generation.py`)
**Class: `GeminiGenerator`**
```python
- API: google.generativeai (Gemini Flash Lite)
- Prompt Template:
  """
  Context from retrieved documents:
  {context}
  
  Question: {question}
  
  Instructions:
  1. Answer using the information from the context
  2. Cite sources: (Source: [Document Name], Page [X])
  3. If context lacks info, state clearly
  4. Be precise and factual
  """
- Safety: Returns error if context is empty
- Debug: Prints context length for troubleshooting
```

### User Interface (`app.py`)
**Streamlit Application Features:**
- **Sidebar:**
  - PDF file uploader (processes on upload)
  - Retrieval mode selector: Sparse/Dense/Hybrid
  - Alpha slider (0.0 - 1.0) for hybrid fusion weight
  
- **Main Chat Interface:**
  - Message history with user/assistant roles
  - Real-time streaming responses
  
- **Debug Expander:**
  - Shows retrieved documents before/after reranking
  - Displays: Document content preview, scores, position changes
  - Format: `#{pre_pos} → #{post_pos} | Score: 0.8542`

---

## 📦 Dependencies

Complete list from `requirements.txt`:

```text
langchain                  # Core LangChain framework
langchain-core            # LangChain core abstractions (Document, etc.)
langchain-community       # Community integrations (PyPDFLoader)
langchain-text-splitters  # Text splitting utilities
langchain-google-genai    # Google Gemini integration
sentence-transformers     # Embedding models and cross-encoders
faiss-cpu                 # Vector similarity search (CPU version)
rank_bm25                 # BM25 sparse retrieval algorithm
pypdf                     # PDF parsing library
streamlit                 # Web application framework
python-dotenv             # Environment variable management
google-generativeai       # Google Gemini API client
scikit-learn             # Machine learning utilities (normalization)
bert-score               # BERTScore evaluation metric
evaluate                 # HuggingFace evaluation library
numpy                    # Numerical computing
```

**Installation:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/sriharsha8991/SHARP.git
cd SHARP

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Google API key
# GOOGLE_API_KEY=your_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Add PDF Documents
Place your scientific PDFs in the `data/` folder:
```bash
SHARP/data/
    ├── paper1.pdf
    ├── paper2.pdf
    └── research_doc.pdf
```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### 5. Using the Interface
1. **Upload PDFs:** Click "Upload PDF files" in the sidebar
2. **Select Retrieval Mode:** Choose Sparse/Dense/Hybrid
3. **Adjust Alpha (Hybrid only):** Control dense/sparse weight (0.0 - 1.0)
4. **Ask Questions:** Type your query in the chat input
5. **View Debug Info:** Expand "Debug: Retrieved Context" to see retrieval details

---

## 🧪 Evaluation Framework

The `src/evaluation.py` module provides comprehensive metrics for assessing retrieval and generation quality:

### Retrieval Metrics
```python
class RetrievalEvaluator:
    # Measures ranking quality
    - ndcg_at_k(relevance_scores, k=10)
    
    # Measures coverage
    - recall_at_k(retrieved_docs, ground_truth, k=10)
```

### Generation Metrics
```python
class GenerationEvaluator:
    # Semantic similarity to reference answer
    - bert_score(generated, reference)
    
    # LLM-as-a-judge scoring (future)
    - llm_judge_score(question, context, answer)
```

### Evaluation Workflow
1. Prepare ground truth dataset (JSON format):
   ```json
   {
     "questions": ["What is...?"],
     "relevant_docs": [[1, 5, 12]],
     "reference_answers": ["The answer is..."]
   }
   ```
2. Run evaluation across all retrieval modes
3. Compare NDCG@10, Recall@10, BERTScore
4. Analyze failure cases using debug visualizations

---

## 📚 Academic Alignment: CSC 785 Final Project

### Topic Selection: **System 1 - IR + LLMs / Generative Retrieval**

#### 🎯 Quick Alignment Reference

| Course Topic Requirement | S.H.A.R.P. Implementation | Evidence |
|-------------------------|---------------------------|----------|
| **Integrating LLMs with Retrieval Pipelines (RAG)** | Complete RAG architecture with 3 retrieval modes + LLM | `src/retrieval.py`, `src/generation.py`, `app.py` |
| **Prompt Engineering for RAG** | Citation-enforcing prompt templates with structured instructions | `GeminiGenerator.DEFAULT_PROMPT_TEMPLATE` |
| **LLM-as-Index Comparison** | Baseline mode for direct LLM queries (no retrieval) | `generate_without_retrieval()` method |
| **Evaluating Hallucination & Factual Consistency** | Citation tracking, BERTScore, context verification, debug visualization | `src/evaluation.py`, debug expander in UI |

---

S.H.A.R.P. directly addresses all four key aspects of the chosen topic from the course project guidelines:

#### 1. ✅ Integrating LLMs with Retrieval Pipelines (RAG)

**Implementation:**
- **Complete RAG Architecture:** S.H.A.R.P. implements a full Retrieval-Augmented Generation pipeline
- **LLM Integration:** Google Gemini Flash Lite (`models/gemini-flash-lite-latest`) for generation
- **Multi-Strategy Retrieval:** Three retrieval modes (Sparse, Dense, Hybrid) feed context to the LLM
- **Cross-Encoder Re-ranking:** Two-stage retrieval ensures highest-quality context reaches the LLM

**Code Evidence:**
```python
# From src/generation.py
class GeminiGenerator:
    """Integrates Google Gemini with retrieved context"""
    
    def generate(self, query: str, documents: List[Document]) -> str:
        # Constructs prompt with retrieved context
        context = self._format_context(documents)
        prompt = self.prompt_template.format(
            context=context,
            question=query
        )
        # Queries LLM with retrieval-augmented prompt
        response = self.model.generate_content(prompt)
```

**Novel Contributions:**
- Comparative analysis of sparse vs. dense vs. hybrid retrieval impact on generation quality
- Debug visualization showing which retrieved documents influence LLM responses
- Metadata preservation through entire pipeline (source citations in LLM output)

---

#### 2. ✅ Prompt Engineering for Retrieval-Augmented Generation

**Implementation:**
- **Citation-Enforcing Prompt Template:** Designed to minimize hallucinations
- **Structured Instructions:** Multi-step prompt guides LLM behavior
- **Fallback Handling:** Explicit instructions for insufficient context scenarios

**Prompt Design:**
```python
DEFAULT_PROMPT_TEMPLATE = """You are an expert assistant that answers questions based on the provided context from scientific documents.

Context from retrieved documents:
{context}

Question: {question}

Instructions:
1. Answer using the information from the context above.
2. Cite sources using format: (Source: [Document Name], Page [X])
3. If context contains relevant information, use it to provide a helpful answer.
4. If context truly doesn't contain any relevant information, state clearly.
5. Be precise and factual. Summarize key points from the documents.
6. If there are step-by-step procedures, list them clearly.

Answer:"""
```

**Research Contribution:**
- Investigates prompt strategies that enforce citation while maintaining answer quality
- Balances strictness (preventing hallucination) with flexibility (allowing useful responses)
- Implements safety mechanisms (empty context detection, error handling)

---

#### 3. ✅ LLM-as-Index: Direct Question Answering

**Implementation:**
While S.H.A.R.P. uses a retriever-then-generate approach (standard RAG), it includes evaluation capabilities for comparing:
- **Traditional RAG:** Retrieve → Rerank → Generate (current implementation)
- **Direct LLM:** Query LLM without retrieval (baseline for comparison)
- **Hybrid Approaches:** Varying amounts of context (evaluating when retrieval helps vs. hurts)

**Code Support:**
```python
# From src/generation.py - Can operate with or without retrieval
def generate_without_retrieval(self, query: str) -> str:
    """Direct LLM query without RAG context (baseline)"""
    prompt = f"Question: {query}\n\nAnswer:"
    return self.model.generate_content(prompt).text
```

**Research Angle:**
- Empirical comparison: When does retrieval improve over pure LLM generation?
- Cost-benefit analysis: Computational overhead of retrieval vs. answer quality gains
- Domain-specific evaluation: Scientific PDFs require grounding (hypothesis)

---

#### 4. ✅ Evaluating Hallucination and Factual Consistency

**Implementation:**
- **Citation Tracking:** Every claim must reference source document and page number
- **Context Verification:** Debug mode shows exact text chunks fed to LLM
- **Multiple Evaluation Metrics:** NDCG, Recall, BERTScore for retrieval quality
- **Factual Grounding:** LLM answers constrained to retrieved context

**Evaluation Framework (`src/evaluation.py`):**
```python
class RetrievalEvaluator:
    def ndcg_at_k(self, relevance_scores, k=10):
        """Measures ranking quality - better ranking = better context for LLM"""
        
    def recall_at_k(self, retrieved_docs, ground_truth, k=10):
        """Measures if relevant docs reach the LLM"""

class GenerationEvaluator:
    def bert_score(self, generated, reference):
        """Semantic similarity to ground truth answers"""
        
    def citation_accuracy(self, answer: str, documents: List[Document]):
        """Verifies citations match actual document content (NEW)"""
```

**Hallucination Detection Strategy:**
1. **Pre-generation:** Ensure high-quality context via re-ranking
2. **During generation:** Prompt enforces citation requirements
3. **Post-generation:** Verify cited content exists in source documents
4. **Comparative analysis:** BERTScore against reference answers

**Research Contributions:**
- Quantifies how retrieval strategy (sparse/dense/hybrid) affects hallucination rates
- Measures impact of re-ranking on factual consistency
- Analyzes trade-offs between answer completeness and citation accuracy

---

### 📋 System Development Deliverables (Project Option 1)

✅ **Domain-Specific Corpus:** Scientific PDFs (research papers, technical documents)  
✅ **Hybrid Retrieval System:** BM25 (sparse) + FAISS embeddings (dense) with weighted fusion  
✅ **Neural Re-ranking:** Cross-encoder (`ms-marco-MiniLM-L-6-v2`) for precision improvement  
✅ **LLM Integration:** Google Gemini Flash Lite for generation with citation enforcement  
✅ **Evaluation Suite:** NDCG@K, Recall@K, BERTScore for comprehensive assessment  
✅ **Interactive Demo:** Streamlit web application with debug visualization  

---

### 🔬 Research Questions Addressed

**RQ1:** How do different retrieval strategies (sparse, dense, hybrid) affect LLM generation quality in scientific document QA?
- **Hypothesis:** Hybrid retrieval with α=0.5 balances keyword precision (BM25) and semantic understanding (dense embeddings)
- **Variables:** Retrieval mode {Sparse, Dense, Hybrid}, α {0.3, 0.5, 0.7}
- **Metrics:** BERTScore (F1), citation accuracy, user preference

**RQ2:** Does cross-encoder re-ranking reduce hallucinations in LLM-generated answers?
- **Hypothesis:** Re-ranking improves context quality → fewer hallucinations
- **Variables:** Re-ranking {Enabled, Disabled}, Top-K {5, 10, 20}
- **Metrics:** Hallucination rate (manual annotation), citation verification accuracy

**RQ3:** What is the trade-off between citation enforcement and answer usefulness?
- **Hypothesis:** Strict prompts reduce hallucinations but may over-constrain useful extrapolation
- **Variables:** Prompt strictness {Strict, Balanced, Lenient}
- **Metrics:** Citation coverage, answer completeness, user satisfaction

**RQ4:** Can retrieval metadata (source, page number) improve answer verifiability?
- **Hypothesis:** Metadata-preserving pipeline enables fact-checking and increases user trust
- **Variables:** Citation format, metadata granularity
- **Metrics:** Verification time, trust rating (user study)

---

### 🧪 Experimental Design

**Datasets:**
- **Training/Index:** Scientific PDFs from arXiv, research papers, technical documentation
- **Evaluation:** Custom question set with ground truth answers and relevance judgments

**Baselines:**
1. **BM25-only:** Sparse retrieval + Gemini (lexical baseline)
2. **Dense-only:** FAISS semantic search + Gemini (neural baseline)
3. **Direct LLM:** Gemini without retrieval (no-RAG baseline)

**Proposed System Variants:**
- **S.H.A.R.P.-Hybrid:** α-tunable hybrid + re-ranking + citation-enforced generation
- **S.H.A.R.P.-NoRerank:** Hybrid retrieval without cross-encoder (ablation study)
- **S.H.A.R.P.-Adaptive:** Dynamic α selection based on query type

**Evaluation Metrics:**

| Category | Metric | Purpose |
|----------|--------|---------|
| **Retrieval Quality** | NDCG@10, Recall@10, MRR | Measures ranking effectiveness |
| **Generation Quality** | BERTScore (F1), ROUGE-L | Semantic similarity to reference |
| **Hallucination** | Citation accuracy, manual annotation | Factual consistency |
| **User Experience** | Preference ranking, trust score | Real-world utility |
| **Efficiency** | Latency (ms), throughput (q/s) | Scalability assessment |

**Experimental Procedure:**
1. Index corpus with all three retrieval methods
2. Run each baseline and S.H.A.R.P. variant on test questions
3. Measure retrieval quality (NDCG, Recall) before generation
4. Generate answers with LLM using retrieved context
5. Evaluate generation quality (BERTScore, ROUGE-L)
6. Manually annotate hallucinations and verify citations
7. Conduct user study for preference and trust ratings
8. Statistical analysis (paired t-tests, effect sizes)

---

### 📊 Expected Contributions to IR Field

1. **Empirical Evidence:** Quantifies when hybrid retrieval outperforms sparse/dense alone in RAG systems
2. **Re-ranking Impact:** Demonstrates measurable reduction in hallucinations through cross-encoder re-ranking
3. **Prompt Engineering:** Provides reusable citation-enforcing prompt templates for scientific QA
4. **Evaluation Framework:** Combines retrieval metrics (NDCG) with generation metrics (BERTScore) for end-to-end RAG assessment
5. **Open-Source Implementation:** Fully documented, reproducible system for research community

---

### 📝 Technical Report Outline (IEEE Format)

**1. Abstract**
- Problem: Hallucinations in LLM-based scientific QA
- Solution: S.H.A.R.P. hybrid RAG with citation enforcement
- Results: Hybrid (α=0.5) + re-ranking reduces hallucinations by X% vs. baselines

**2. Introduction**
- Motivation: Need for factually grounded answers in scientific domains
- Challenges: Balancing recall (BM25) with precision (embeddings), hallucination control
- Contributions: Multi-strategy RAG, citation tracking, comprehensive evaluation

**3. Related Work**
- Classical IR: BM25, TF-IDF
- Neural IR: Dense retrieval (DPR, ColBERT), re-ranking (cross-encoders)
- RAG systems: LangChain, LlamaIndex, existing implementations
- Hallucination detection: Citation verification, factual consistency metrics

**4. System Design**
- Architecture diagram (5 stages: Ingestion → Indexing → Retrieval → Re-ranking → Generation)
- Implementation details for each component (from Technical Specifications section)
- Design decisions: Why hybrid? Why cross-encoder? Why citation-enforcing prompts?

**5. Experiments**
- Dataset description
- Baseline configurations
- Evaluation metrics
- Experimental procedure

**6. Results**
- Retrieval quality comparison (Table: NDCG@10, Recall@10 across modes)
- Generation quality comparison (Table: BERTScore, ROUGE-L, hallucination rate)
- Ablation studies (re-ranking impact, α sensitivity analysis)
- User study findings (preference rankings, trust scores)

**7. Discussion**
- Key findings: When does hybrid outperform? Impact of re-ranking?
- Limitations: Domain-specific (scientific PDFs), API costs, latency
- Failure analysis: What questions still cause hallucinations?
- Future work: Multi-turn conversations, query expansion, ensemble re-ranking

**8. Conclusion**
- Summary: S.H.A.R.P. demonstrates hybrid RAG + re-ranking reduces hallucinations
- Impact: Provides reusable framework for factual scientific QA
- Next steps: Extend to other domains (legal, medical), scale to larger corpora

**9. References**
- LangChain, Sentence Transformers, FAISS documentation
- Key papers: DPR (Karpukhin et al.), ColBERT (Khattab & Zaharia), RAG (Lewis et al.)
- Evaluation benchmarks: MS MARCO, BEIR

---

### 🎥 Presentation Structure (10-15 minutes)

**Slide 1: Title & Team** (30s)
- Project name, course, team members

**Slide 2: Motivation** (1 min)
- Problem: Hallucinations in scientific QA
- Example: LLM making up citations

**Slide 3: Research Questions** (1 min)
- RQ1-RQ4 from above

**Slide 4: System Architecture** (2 min)
- Diagram showing 5-stage pipeline
- Highlight: 3 retrieval modes, re-ranking, citation enforcement

**Slide 5: Technical Implementation** (2 min)
- Key technologies: LangChain, FAISS, Gemini
- Code snippet: Hybrid retrieval formula
- Code snippet: Citation-enforcing prompt

**Slide 6: Experimental Setup** (1 min)
- Datasets, baselines, metrics

**Slide 7: Results - Retrieval Quality** (2 min)
- Table/chart: NDCG@10, Recall@10 comparison
- Finding: Hybrid (α=0.5) best overall

**Slide 8: Results - Generation Quality** (2 min)
- Table/chart: BERTScore, hallucination rate
- Finding: Re-ranking reduces hallucinations by X%

**Slide 9: Demo** (2 min)
- Live/recorded Streamlit UI walkthrough
- Upload PDF → Ask question → Show debug view with citations

**Slide 10: Conclusion & Future Work** (1 min)
- Key takeaways
- Limitations and next steps

**Slide 11: Q&A** (remaining time)

---

### 📦 Project Deliverables Checklist

**Technical Report (IEEE Format):**
- [ ] Abstract (150-200 words)
- [ ] Introduction with motivation
- [ ] Related work survey (10+ citations)
- [ ] System design with architecture diagrams
- [ ] Experimental methodology
- [ ] Results with tables/figures
- [ ] Discussion and analysis
- [ ] Conclusion and future work
- [ ] References (IEEE format)

**Code Submission:**
- [x] `src/` directory with all modules
- [x] `app.py` Streamlit application
- [x] `requirements.txt` with dependencies
- [x] `.env.example` for configuration
- [x] `README.md` with setup instructions
- [ ] Jupyter notebook with evaluation experiments
- [ ] Sample PDFs in `data/` for testing
- [ ] Pre-computed indices in `indices/` (optional)

**Video Presentation:**
- [ ] 10-15 minute recorded presentation
- [ ] All team members present
- [ ] Screen recording of slides + demo
- [ ] .mp4 format upload to D2L

**Verification:**
- [ ] Code runs without errors
- [ ] Results in paper match code output
- [ ] No plagiarism (proper citations)
- [ ] Single team submission to D2L

---

### Research Questions Addressed

1. **RQ1:** How do different retrieval strategies (sparse, dense, hybrid) affect LLM generation quality in scientific document QA?
   - **Hypothesis:** Hybrid retrieval with α=0.5 balances keyword precision (BM25) and semantic understanding (dense embeddings)
   
2. **RQ2:** Does cross-encoder re-ranking reduce hallucinations in LLM-generated answers?
   - **Hypothesis:** Re-ranking improves context quality → fewer hallucinations
   
3. **RQ3:** What is the trade-off between citation enforcement and answer usefulness?
   - **Hypothesis:** Strict prompts reduce hallucinations but may over-constrain useful extrapolation
   
4. **RQ4:** Can retrieval metadata (source, page number) improve answer verifiability?
   - **Hypothesis:** Metadata-preserving pipeline enables fact-checking and trust

---

### Experimental Design

**Datasets:**
- Scientific PDFs (arXiv, research papers, technical documentation)
- User-generated question set with ground truth answers

**Baselines:**
1. BM25 only + Gemini
2. Dense retrieval only + Gemini
3. Direct Gemini (no retrieval)

**Proposed System:**
- Hybrid (α-tunable) + Cross-Encoder Re-ranking + Citation-Enforced Gemini

**Metrics:**
- **Retrieval Quality:** NDCG@10, Recall@10, MRR
- **Generation Quality:** BERTScore (F1), ROUGE-L, citation accuracy
- **Hallucination Rate:** Manual annotation + automated citation verification
- **User Study:** Preference ranking (answer quality, trustworthiness)

**Variables:**
- Retrieval mode: {Sparse, Dense, Hybrid}
- Alpha weight: {0.3, 0.5, 0.7}
- Re-ranking: {Enabled, Disabled}
- Top-K: {5, 10, 20}

---

## 🐛 Troubleshooting

### Common Issues

**1. "GOOGLE_API_KEY not found"**
- Solution: Create `.env` file with `GOOGLE_API_KEY=your_key`
- Verify: Check that `.env` is in the project root

**2. "Cannot find sufficient information..."**
- Cause: Empty or irrelevant context reaching the LLM
- Debug: Check "Debug: Retrieved Context" expander
- Solutions:
  - Try different retrieval modes (Hybrid often works best)
  - Adjust alpha parameter (try 0.3-0.7 range)
  - Verify PDFs contain searchable text (not scanned images)

**3. Reranking not improving results**
- Check rerank scores in debug view
- Scores should be positive (higher = more relevant)
- If scores seem inverted, verify cross-encoder model is loaded correctly

**4. Slow performance**
- First run is slower (model downloads)
- Consider using GPU version: `faiss-gpu` instead of `faiss-cpu`
- Reduce `TOP_K` in `config.py` for faster retrieval

**5. Out of memory errors**
- Reduce `CHUNK_SIZE` in `config.py`
- Process fewer PDFs at once
- Use smaller embedding model (though less accurate)

---

## 📊 Performance Characteristics

### Model Sizes
- **Embedding Model** (`all-mpnet-base-v2`): ~420MB
- **Reranker Model** (`ms-marco-MiniLM-L-6-v2`): ~80MB
- **FAISS Index**: ~3MB per 1000 chunks

### Speed Benchmarks (CPU, 10 documents)
- Sparse Retrieval: ~50ms
- Dense Retrieval: ~100ms
- Hybrid Retrieval: ~150ms
- Re-ranking (10 docs): ~200ms
- Generation: ~1-3s (depends on answer length)

**Total Pipeline Latency: ~2-4 seconds**

---

## 🔬 Research Background

This implementation is based on state-of-the-art RAG research:

1. **Hybrid Retrieval**: Combines lexical (BM25) and semantic (dense) search for better recall
2. **Cross-Encoder Re-ranking**: Two-stage retrieval improves precision vs. single-stage
3. **Citation Enforcement**: Reduces hallucinations by grounding answers in source documents
4. **Min-Max Normalization**: Critical for combining scores from different retrieval methods

### Key Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)
- "RankT5: Fine-Tuning T5 for Text Ranking" (Zhuang et al., 2022)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for more document formats (DOCX, HTML, Markdown)
- [ ] Implement query expansion techniques
- [ ] Add ensemble reranking with multiple cross-encoders
- [ ] Integrate advanced chunking strategies (semantic, sliding window)
- [ ] Add caching layer for faster repeated queries
- [ ] Implement conversation memory for multi-turn dialogues

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🙏 Acknowledgments

- **LangChain** for document processing framework
- **Sentence Transformers** for embedding and reranking models
- **Google** for Gemini API access
- **FAISS** for efficient vector search
- **Streamlit** for rapid UI development