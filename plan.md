Hybrid Retrieval Augmented Generation System for Information Extraction 
Final Project topic : Information Storage and Retrieval  
Students involved: 
1. 
2. 
3. 
4. 
Project Type : System Development / application development 
Focus Area s: / Generative Retrieval Models / IR + LLMs (Topic 4) 
Date: 
Abstract 
Search systems can fetch only relevant piece of information, but cannot structurise the response as per the users query and misses capturing of semantic meaning. To address this gap, our project focuses on developing a Hybrid Retrieval Augmented Generation (RAG) framework that brings together the strengths of different retrieval methods 
We are trying to make a system combines BM25 for keyword-level retrieval, SentenceTransformer embeddings for semantic matching, and an LLM (Gemini) for context-aware synthesis.The System will be compared based on Sparse Vs Dense Vs hybrid retrieval techniques used 
Our retrieval quality will be using NDCG@10, Recall@k, and MAP, while answering quality will be evaluated using BERTScore, ROUGE, and factual consistency metrics. 
I. Introduction and Motivation 
We know that information Retrieval (IR) traditionally focuses on matching documents to queries through lexical overlap (BM25, TF-IDF). While efficient, such methods lack semantic understanding and cannot automatically summarize or interpret results.With the advent of neural and generative models, Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm that unites retrieval precision with LLM reasoning. Yet, most RAG systems focus on open-domain QA rather than generating structured, actionable sequences—the critical missing link between “information” and “instruction.” 
Our project aims to build a domain-adaptive hybrid RAG pipeline capable of extracting and synthesizing step-wise actions from Scentific articles and documentations. And we will evaluate
how combining sparse + dense retrieval with LLM-based re-ranking can improve factual accuracy and minimize hallucination in procedural outputs. 
II . System Design and Methodology: 
A. Overview 
In our project, the proposed architecture (Figure 1) follows a three-stage retrieval and generation pipeline: 
1. 
Indexing & embedding, 
2. 
Hybrid retrieval & re-ranking, and 
3. 
LLM-based answer generation . 
B. Indexing & Embedding Pipeline : 
1. Corpus data: 
We are selecting Multiple documents which are PDFs and text documents related to scientific and research articles 
2. Chunking: 
Each Q&A thread is pre-processed into smaller text segments (~256 tokens with 50 overlap) to preserve context during retrieval and an additional metadata document name and page numbers will be added programatically 
3. Embedding Generation: 
Use SentenceTransformers (all-mpnet-base-v2 or 
multi-qa-MiniLM-L6-cos-v1) to generate semantic embeddings. 
4. Vector Storage: 
Store embeddings in a vector database (Qdrant/ FAISS / Chroma) optimized for Approximate Nearest Neighbor (ANN) or HNSW (for Qdrant ) search, enabling fast and efficient similarity queries. 
C. Hybrid Retrieval & Re-ranking : 
The below are some of the hybrid retrieval and reranking techniques we are planning to use, which are as follows: 
1. Sparse Retrieval: 
Employ BM25 (via PyTerrier or Elasticsearch) to retrieve top K candidates based on lexical similarity. 
2. Dense Retrieval: 
Conducts ANN search/ HNSW graph based search using the vector store(FAISS or Qdrant) to retrieve top K semantically similar chunks. 
3. Fusion Layer: 
Combines sparse and dense scores using a weighted fusion method (e.g., linear interpolation) or learning-to-rank model (LambdaMART),this the fusion reranking layer. 4. Neural Re-ranking: 
Uses a cross-encoder model (ms-marco-MiniLM-L-6-v2) to refine top K′ passages,
maximizing relevance and context alignment, this helps getting the most relevant chunks. 
D. Question and Answer Generation: 
Augmentation is the key to generate the outputs in the desired format : Augmentation is nothing combining the retrieved context and then using the LLM capabilities to generate the final response in simple terms, below are some of the important steps required 
1. Prompt Construction: 
Based on the top chunks retrieved we will integrate the retrieved K′ passages into a structured system prompt, instructing the LLM to produce an evidence-based answer: “Answer the user’s question using only the provided context. By grounding to retrieve context chunks 
2. LLM Integration: 
We are going to utilise the LLMs Via APIs, we might use an LLM (Gemini) to generate coherent, grounded answers.The prompt explicitly instructs the model to avoid unsupported claims and cite context sources. 
3. Post-processing: 
We are aiming to format outputs into a Q&A style pair (User asks Question → LLM Generates an answer → with citation) so that every response is grounded to the source it retrieved and will have some evidence. 
E. Proposed Workflow (Figure 1) (high level Overview of the architecture)

III. Evaluation Strategy 
The below are the evaluation strategies we are planning to follow to test our approach. 
A. Retrieval Effectiveness: 
Baseline 
Model 
Metrics
Sparse 
Vectors
BM25 (PyTerrier ) 
MAP & MRR
Dense 
Vectors
SBERT (all-mpnet 
base-v2)
NDCG @10 & 
Recall@k
Hybrid 
approach
Weighted Fusion technique 
Combined MAP / Recall



We are going to make an evaluation test setup with manual relevance annotations on ≈ 100 query-chunk pairs will be used for ground-truth validation, mostly in the form of CSV/JSON. 
B. Generative Evaluation 
Aspects 
Metrics & Method
Factual Consistency 
BERT Score for retrieved context
Hallucination Rates 
LLM acting as judge
Actionability / 
Completeness
LLM as Judge scoring (1–5 scale)
Latency Trade-off 
Average retrieval + generation time per query



Comparision between actual Ground truth responses and LLM generated responses 
IV. Expected Contributions: 
By the end of this project we are trying to produce the following outcomes : 
1. A reproducible hybrid RAG framework integrating sparse and dense retrieval with proper LLM reasoning. 
2. Quantitative results showing improved factual & response generation areas with llm. 3. Comparing sparse, dense, and hybrid retrieval within domain-related data with evaluation criterias mentioned above. 
Points to note: Models mentioned in the document are subjected to Change in future while implementating depending up on the resource requirements / constraints .
