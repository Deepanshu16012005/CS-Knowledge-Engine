# 📚 CS Knowledge Base — Hybrid RAG Q&A System

A production-grade **Hybrid Retrieval-Augmented Generation (RAG)** pipeline for answering questions on **Data Structures & Algorithms (DSA)** and **Operating Systems (OS)** from personal PDF notes.

This system combines **dense + sparse vector search**, **Cohere reranking**, and **Groq LLM generation** to deliver highly accurate, context-grounded answers.

---

## ✨ Key Features

- 🔀 **Hybrid Search** — Combines dense vectors (Gemini semantic embeddings) + sparse vectors (BM25 keyword matching) for best-of-both retrieval
- ⚖️ **Dynamic Alpha Weighting** — Automatically adjusts the balance between keyword vs. semantic search based on query length
- 🏆 **Cohere Reranking** — Re-scores retrieved chunks using `rerank-v3.5` for maximum relevance before generation
- 💬 **Multi-turn Chat** — Maintains last 2 turns of history with automatic query reformulation for follow-up questions
- 🚫 **No Hallucination** — Strict system prompt ensures answers come only from your uploaded PDF

---

## 📁 Project Structure

```
CS-KNOWLEDGEBASE/
├── pdf/
│   └── Dsa.pdf                         # Source PDF notes
├── sparse_vectors/
│   ├── vocab_save.py                   # BM25 training & saving utility
│   └── bm_25_vocab_params.json         # Trained BM25 vocabulary (auto-generated)
├── prompts/
│   ├── system.txt                      # System prompt for Groq answer generation
│   └── query_formulator.txt            # System prompt for query reformulation
├── rag_env/                            # Virtual environment (local only, not committed)
├── .env                                # API keys (not committed)
├── .env.example                        # API key template
├── .gitignore
├── ingest_data.py                      # Chunk, embed (hybrid), upload to Pinecone
├── rag.py                              # Hybrid retrieval + Cohere reranking + Groq generation
├── reranker.py                         # Cohere reranking logic
├── retrieve.py                         # Interactive chat loop with query reformulation
└── requirements.txt                    # Python dependencies
```

---

## ⚙️ How It Works

### Step 1 — Ingestion *(Run Once)*

```
📄 PDF (Dsa.pdf)
     │
     ▼
PyPDFLoader loads the PDF
     │
     ▼
RecursiveCharacterTextSplitter
  chunk_size=2200, chunk_overlap=150
     │
     ▼
BM25Encoder.fit() trains on raw page text
Saves vocabulary → bm_25_vocab_params.json
     │
     ▼
For each batch of 10 chunks:
  ├── Gemini embeds → Dense Vector  (semantic meaning)
  ├── BM25 encodes  → Sparse Vector (keyword signals)
  └── Pinecone upsert: { id, values, sparse_values, metadata }
```

### Step 2 — Query & Answer

```
User Question
     │
     ▼
retrieve.py
  ├── Loads last 2 turns of chat history
  └── Groq reformulates vague/follow-up into standalone query

     │
     ▼
rag.py :: get_rag_answer()
  ├── Gemini embeds query     → Dense Vector
  ├── BM25 encodes query      → Sparse Vector
  │
  ├── Dynamic Alpha Weighting:
  │     ≤ 3 words  → alpha=0.3 (trust keywords more)
  │     > 3 words  → alpha=0.7 (trust semantics more)
  │
  ├── Pinecone hybrid query (top_k=3)
  │
  ├── Cohere rerank-v3.5 reranks results
  │     (fallback: raw Pinecone results if Cohere fails)
  │
  └── Groq llama-3.1-8b-instant generates final answer

     │
     ▼
💬 Grounded Answer with Page & File Source
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **LangChain** | Orchestration & prompt management |
| **Google Gemini** (`gemini-embedding-001`) | Dense vector embeddings |
| **BM25Encoder** (`pinecone-text`) | Sparse vector generation (keyword-based) |
| **Pinecone** | Hybrid vector database (dense + sparse) |
| **Cohere** (`rerank-v3.5`) | Cross-encoder reranking of retrieved chunks |
| **Groq** (`llama-3.1-8b-instant`) | Fast LLM for answer generation & query reformulation |
| **PyPDFLoader** | PDF ingestion |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone "https://github.com/Deepanshu16012005/CS-Knowledge-Engine.git"
cd CS-Knowledge-Engine
```

### 2. Create & Activate Virtual Environment

```bash
python3.11 -m venv rag_env

# Windows
rag_env\Scripts\activate

# Mac/Linux
source rag_env/bin/activate
```

> `rag_env/` is in `.gitignore`. Every user creates their own locally.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Fill in `.env`:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
COHERE_API_KEY=your_cohere_api_key
```

> Get your keys from:
> - Gemini → https://aistudio.google.com/app/apikey
> - Groq → https://console.groq.com/keys
> - Pinecone → https://app.pinecone.io/
> - Cohere → https://dashboard.cohere.com/api-keys

### 5. Set Up Pinecone Index

In your Pinecone dashboard, create an index with:
- **Index name:** `ragproject`
- **Dimensions:** `3072` *(Gemini gemini-embedding-001 output size)*
- **Metric:** `dotproduct` *(required for hybrid search)*

### 6. Add Your PDF

Place your PDF inside `pdf/` and verify the path in `ingest_data.py`:

```python
pdf_path = "./pdf/Dsa.pdf"   # line 13
```

### 7. Ingest Data *(Run Once)*

```bash
python ingest_data.py
```

This will:
1. Load and chunk your PDF
2. Train & save the BM25 vocabulary to `sparse_vectors/bm_25_vocab_params.json`
3. Upload hybrid vectors (dense + sparse) to Pinecone in batches of 10

> ⏳ Uploads pause 10 seconds between batches to respect Gemini's rate limits.

### 8. Start Chatting

```bash
python retrieve.py
```

---

## 💬 Chat Commands

| Input | Action |
|-------|--------|
| Any question | Reformulates & answers using hybrid RAG |
| `clear` | Clears chat history for this session |
| `quit` | Exits the program |

**Example session:**
```
What would you like to search for?
> What is a Binary Search Tree?

Formulated Query: What is a Binary Search Tree?
Got Both Sparse and Dense Vectors From Database
Reranking using COHERE
Ranking successful
Generating answer using AI...

Answer By AI:
A Binary Search Tree (BST) is...
--- (File: Dsa.pdf | Page: 42) ---
```

---

## 🔁 Retrieval Pipeline Detail

### Hybrid Search + Dynamic Alpha

For every query, both a **dense** (semantic) and **sparse** (keyword) vector are generated. Their contributions are weighted by **alpha**, which adapts to query length:

```
Short query (≤ 3 words)  →  alpha = 0.3
  Dense weight  = 0.3  (semantic meaning)
  Sparse weight = 0.7  (keyword matching)

Long query (> 3 words)   →  alpha = 0.7
  Dense weight  = 0.7  (semantic meaning)
  Sparse weight = 0.3  (keyword matching)
```

This means a short query like `"mutex"` relies more on keyword matching, while `"explain how deadlock is avoided in OS"` relies more on semantic understanding.

### Cohere Reranking

After Pinecone returns the top 3 hybrid matches, **Cohere `rerank-v3.5`** re-scores them using a cross-encoder model — which reads both the query and each chunk together for much more accurate relevance scoring. If Cohere fails, the system gracefully falls back to raw Pinecone results.

---

## 🧠 Design Decisions

| Decision | Reason |
|----------|--------|
| Hybrid search (dense + sparse) | Dense alone misses exact keyword matches; sparse alone misses semantic meaning. Hybrid gets both. |
| Dynamic alpha weighting | Short queries benefit from keyword precision; long queries benefit from semantic understanding |
| Cohere reranking after retrieval | Vector similarity ≠ true relevance. Cross-encoder reranking significantly improves answer quality |
| `chunk_size=2200, overlap=150` | Captures full concept explanations; overlap prevents context loss at boundaries |
| `top_k=3` chunks | Balances context richness vs. prompt token size |
| `temperature=0.2` for generation | Low creativity for factual, grounded answers |
| `temperature=0.0` for reformulator | Fully deterministic query rewriting |
| Last 2 turns of history | Keeps token usage low while handling follow-up questions |
| BM25 trained on your notes | Learns domain-specific vocabulary (e.g., "mutex", "AVL", "semaphore") for better keyword matching |
| Prompts in `.txt` files | Keeps code clean; prompts can be edited without touching Python files |

---

## ⚠️ Notes

- Run `ingest_data.py` only **once** (or when your PDF changes). Re-running will overwrite existing Pinecone vectors.
- The BM25 vocabulary (`bm_25_vocab_params.json`) must be present before running `rag.py` or `retrieve.py`.
- The system will **only answer from your PDF** — if the answer isn't there, it says so explicitly.

---

## ⚙️ Work in Progress

These are planned enhancements that are not yet implemented:

### 1. 🗂️ Golden Dataset & Offline Evaluation
Manually curate 50–200 question-answer pairs based strictly on the DSA and OS notes, where each answer is human-verified. This **golden dataset** will serve as a ground truth benchmark to measure how well the RAG pipeline performs over time.

**Blocked by:** Gemini API rate limits during bulk answer generation. Will be built incrementally.

### 2. 📊 Automated Evaluation with Ragas
Integrate [Ragas](https://github.com/explodinggradients/ragas) to run offline evaluations against the golden dataset. Key metrics to track:

- **Faithfulness** — Is the generated answer grounded in the retrieved chunks?
- **Answer Relevancy** — Does the answer actually address the question asked?
- **Context Precision** — Are the retrieved chunks relevant to the question?
- **Context Recall** — Are all necessary facts present in the retrieved context?

### 3. ⚙️ CI/CD Pipeline with GitHub Actions
Set up a GitHub Actions workflow so that every push to the repository automatically runs the Ragas evaluation script. If accuracy drops below a defined threshold, the build fails — preventing regressions from sneaking in through prompt changes or code updates.

```yaml
# Planned: .github/workflows/eval.yml
on: [push]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - run: python eval/run_ragas.py
      - run: python eval/check_threshold.py  # Fails build if score drops
```

### 4. 🌐 Web UI (Streamlit / Gradio)
Replace the current terminal-based chat loop with a proper web interface so the system is accessible without running Python scripts manually. The UI will include a chat window, query history, and source citations (page number + file) displayed alongside each answer.

**Candidates:**
- [Streamlit](https://streamlit.io/) — Quick to build, great for demos and portfolios
- [Gradio](https://gradio.app/) — Easy chat interface with built-in history support

---

## 📄 License

This project is open-source. Feel free to use and modify.