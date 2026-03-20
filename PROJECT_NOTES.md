# 🧠 CS Knowledge Base — Hybrid RAG System: Complete Notes

> Covers: Code explanations, keyword meanings, data flow, and concept Q&A

---

## 📁 Table of Contents

1. [File-by-File Code Breakdown](#1-file-by-file-code-breakdown)
   - [vocab_save.py](#-vocab_savepy)
   - [ingest_data.py](#-ingest_datapy)
   - [rag.py](#-ragpy)
   - [reranker.py](#-rerankerpy)
   - [retrieve.py](#-retrievepy)
   - [testing/evaluate.py](#-testingevaluatepy)
2. [Complete Data Flow](#2-complete-data-flow)
   - [Ingestion + Query Flow](#ingestion--query-flow)
   - [Evaluation Flow](#evaluation-flow)
3. [Concept Q&A](#3-concept-qa)
   - [Core RAG Concepts](#core-rag-concepts)
   - [Evaluation & Ragas Concepts](#evaluation--ragas-concepts)

---

# 1. File-by-File Code Breakdown

---

## 📄 `vocab_save.py`

**Purpose:** Train BM25 on your specific PDF text and save the vocabulary to a JSON file. This is a one-time step before ingestion.

```python
from pinecone_text.sparse import BM25Encoder
import json

def train_and_save_bm25(documents_list, save_path="bm25_params.json"):
    bm25 = BM25Encoder()
    bm25.fit(documents_list)       # <-- MOST IMPORTANT LINE
    bm25.dump(save_path)
```

### 🔑 Important Keywords

| Term | Meaning |
|------|---------|
| `BM25Encoder` | A class from `pinecone_text` that implements BM25 — a keyword-scoring algorithm. Converts text into sparse vectors. |
| `.fit(documents_list)` | Trains BM25 on your specific corpus. It learns which words appear frequently and which are rare. Rare words get higher importance scores. |
| `.dump(save_path)` | Saves the trained vocabulary (word frequencies, IDF scores) to a JSON file so you don't have to retrain every time. |
| `documents_list` | A plain Python list of strings — each string is one page of your PDF. |

### 💡 Why `.fit()` matters

Without fitting, BM25 treats all words equally. After fitting on YOUR notes, it learns that words like `"mutex"`, `"semaphore"`, or `"binary tree"` are domain-specific and important — so it gives them higher scores during retrieval.

---

## 📄 `ingest_data.py`

**Purpose:** Load the PDF → chunk it → generate both dense + sparse vectors → upload to Pinecone.

### Block 1 — Imports & Setup

```python
from pinecone_text.sparse import BM25Encoder
from sparse_vectors.vocab_save import train_and_save_bm25
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)
```

**Why native Pinecone client instead of LangChain's `PineconeVectorStore`?**

LangChain's wrapper doesn't support hybrid (sparse + dense) uploads. You need the raw Pinecone client to upload both `values` (dense) and `sparse_values` (sparse) together in one vector.

---

### Block 2 — Load & Chunk the PDF

```python
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=2200,
   chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)
```

### 🔑 Important Keywords

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `chunk_size` | `2200` | Max characters per chunk. Large enough to capture full concept explanations from DSA/OS notes. |
| `chunk_overlap` | `150` | Characters shared between consecutive chunks. Prevents a concept from being cut off at a boundary. |
| `RecursiveCharacterTextSplitter` | — | Splits by paragraph → sentence → word, in that order of preference. Tries to keep logical units together. |

---

### Block 3 — Train & Load BM25

```python
# Used for pages (not chunks) to learn the vocab
text_chunks = [doc.page_content for doc in documents]
train_and_save_bm25(text_chunks, "./sparse_vectors/bm_25_vocab_params.json")

# Now load the saved encoder for use in ingestion
bm25_encoder = BM25Encoder()
bm25_encoder.load("./sparse_vectors/bm_25_vocab_params.json")
```

**Note:** BM25 is trained on raw pages (`documents`) not on split chunks. This gives it a better overall vocabulary of your whole PDF before encoding individual chunks.

---

### Block 4 — The Hybrid Upload Loop (Most Important)

```python
for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    texts = [doc.page_content for doc in batch]

    # Build metadata — include the raw text for retrieval later
    metadatas = []
    for doc in batch:
        m = doc.metadata.copy()
        m["text"] = doc.page_content   # <-- CRITICAL
        metadatas.append(m)

    ids = [f"chunk_{j}" for j in range(i, i + len(batch))]

    # Generate dense vectors (meaning/semantics)
    dense_vectors = embeddings_model.embed_documents(texts)

    # Generate sparse vectors (keywords)
    sparse_vectors = bm25_encoder.encode_documents(texts)

    # Package into Pinecone format
    vectors_to_upload = []
    for j in range(len(batch)):
        vectors_to_upload.append({
            "id": ids[j],
            "values": dense_vectors[j],         # Dense (Gemini)
            "sparse_values": sparse_vectors[j], # Sparse (BM25)
            "metadata": metadatas[j]
        })

    index.upsert(vectors=vectors_to_upload)
    time.sleep(10)  # Gemini rate limit cooldown
```

### 🔑 Important Keywords

| Term | Meaning |
|------|---------|
| `embed_documents(texts)` | Converts a list of text strings into dense float vectors using Gemini. Each vector captures the semantic meaning. |
| `encode_documents(texts)` | Converts text into sparse vectors using BM25. Returns `{indices: [...], values: [...]}` — only non-zero word positions are stored. |
| `metadata["text"]` | Explicitly storing the raw chunk text in Pinecone metadata. Without this, you can't recover the text later during retrieval. |
| `upsert` | "Update or Insert" — if a vector with the same ID exists, it gets overwritten; otherwise it gets inserted. |
| `sparse_values` | A special Pinecone field that holds BM25 sparse vectors alongside the dense embedding. |
| `ids` | Every vector needs a unique string ID. `chunk_0`, `chunk_1`... are used here. |

---

## 📄 `rag.py`

**Purpose:** The core retrieval + generation engine. Given a query, find the most relevant chunks using hybrid search, rerank them, and generate an answer via Groq.

### Block 1 — Connections & Models

```python
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("ragproject")

bm25_encoder = BM25Encoder()
bm25_encoder.load("./sparse_vectors/bm_25_vocab_params.json")

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_retries=2
)
```

**Same BM25 file used in ingestion must be loaded here** — vocabulary must match exactly. If you retrain BM25 on different data and use it here, the sparse vectors won't align with what's stored in Pinecone.

### 🔑 Important Keywords

| Parameter | Meaning |
|-----------|---------|
| `temperature=0.2` | Controls randomness of LLM output. 0.0 = fully deterministic, 1.0 = very creative. 0.2 is low — good for factual answers. |
| `max_retries=2` | If the Groq API fails, automatically retry up to 2 times before raising an error. |

---

### Block 2 — The Alpha Scaling (Hybrid Weighting)

```python
words = query_text.split()
if len(words) <= 3:
    alpha = 0.3  # Short query? Trust keywords more.
else:
    alpha = 0.7  # Long query? Trust meaning more.

scaled_dense = [v * alpha for v in dense_vec]

scaled_sparse = {
    'indices': sparse_vec['indices'],
    'values': [v * (1 - alpha) for v in sparse_vec['values']]
}
```

### 🔑 What is Alpha?

`alpha` is a weight that controls the balance between dense and sparse search.

| Scenario | Alpha | Dense Weight | Sparse Weight | Why |
|----------|-------|-------------|---------------|-----|
| Short query (`<= 3 words`) | 0.3 | 30% | 70% | Short queries are likely keyword lookups — "BFS algorithm", "mutex lock". Keywords matter more. |
| Long query (`> 3 words`) | 0.7 | 70% | 30% | Long queries express meaning — "explain how deadlock occurs in OS". Semantics matter more. |

This is called **dynamic alpha** — the balance shifts based on query length.

---

### Block 3 — Hybrid Query to Pinecone

```python
query_results = index.query(
    vector=scaled_dense,
    sparse_vector=scaled_sparse,
    top_k=3,
    include_metadata=True
)
```

### 🔑 Important Keywords

| Parameter | Meaning |
|-----------|---------|
| `vector` | The dense (Gemini) embedding of the query — scaled by alpha. |
| `sparse_vector` | The BM25 sparse encoding of the query — scaled by (1 - alpha). |
| `top_k=3` | Retrieve the top 3 most similar chunks. These 3 go to the reranker. |
| `include_metadata=True` | Include the stored metadata (page number, source, raw text) in the results. Without this, you only get IDs and scores. |

---

### Block 4 — Reranking + Fallback

```python
try:
    context_text = rerank_pinecone_matches(
        query=query_text,
        pinecone_matches=query_results['matches'],
        top_n=3
    )
except Exception as e:
    # Fallback: format chunks manually without reranking
    context_pieces = []
    for match in query_results['matches']:
        page_num = match.metadata.get('page_label', 'Unknown')
        formatted_chunk = f"--- (Page: {page_num} ...) ---\n{match.metadata.get('text','...')}"
        context_pieces.append(formatted_chunk)
    context_text = "\n\n".join(context_pieces)
```

**Good practice:** Always have a fallback in case the reranker API fails. Here, if Cohere reranking fails, the code still works — it just uses the original Pinecone ordering.

---

### Block 5 — Answer Generation

```python
rag_chain = prompt | groq_llm

response = rag_chain.invoke({
    "context": context_text,
    "question": query_text
})
return response.content
```

### 🔑 Important Keywords

| Term | Meaning |
|------|---------|
| `prompt \| groq_llm` | LangChain **pipe operator**. Chains the prompt template and LLM together. Output of prompt becomes input to LLM. |
| `rag_chain.invoke({...})` | Runs the entire chain with the given variables. The `{context}` and `{question}` placeholders in the prompt get filled. |
| `response.content` | The actual text answer from Groq. The full response object has more metadata, but `.content` is just the text. |

---

## 📄 `reranker.py`

**Purpose:** Take Pinecone's top matches and reorder them using Cohere's reranking model — which is much better at judging relevance than vector similarity alone.

```python
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

def rerank_pinecone_matches(query: str, pinecone_matches: list, top_n: int = 3) -> str:

    # Step 1: Extract just text strings for Cohere
    docs_to_rerank = [match['metadata'].get('text', '') for match in pinecone_matches]

    # Step 2: Call Cohere reranker
    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=docs_to_rerank,
        top_n=top_n
    )

    # Step 3: Reconstruct with metadata using r.index
    for r in rerank_response.results:
        original_match = pinecone_matches[r.index]   # <-- KEY LINE
        meta = original_match['metadata']
        content = meta.get('text', "")
        page = meta.get('page_label', 'Unknown')
        source = os.path.basename(meta.get('source', 'unknown'))
        formatted_chunk = f"\n--- (File: {source} | Page: {page}) ---\n{content}"
```

### 🔑 Important Keywords

| Term | Meaning |
|------|---------|
| `co.rerank()` | Cohere's API call. Sends the query + documents and returns them reordered by relevance score. |
| `model="rerank-v3.5"` | Cohere's reranking model. It's a cross-encoder — it reads query AND document together to score relevance. More accurate than vector similarity. |
| `top_n=3` | Return only the top 3 results after reranking. |
| `r.index` | The original position of this result in the input list. Used to look up the full Pinecone metadata from `pinecone_matches[r.index]`. |
| `os.path.basename(source)` | Extracts just the filename from a full path. `./pdf/Dsa.pdf` → `Dsa.pdf`. Cleaner for display. |

**Why `r.index` is critical:**
Cohere only receives plain text strings. After reranking, `r.index` tells you which original Pinecone match corresponds to this result — so you can pull back the page number, source, and other metadata for the LLM prompt.

---

## 📄 `retrieve.py`

**Purpose:** The user-facing chat loop. Reformulates vague queries using history, then calls the RAG engine.

### Block 1 — Query Reformulator

```python
formulator_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,   # <-- Fully deterministic
    max_retries=2
)

reformulator_chain = reformulator_prompt | formulator_llm
```

**Why `temperature=0.0` here?**
The reformulator should produce the same rewritten query every time for a given input. There's no need for creativity — you want a deterministic, clean standalone question. `0.0` eliminates randomness completely.

---

### Block 2 — Chat Loop with History

```python
history_text = ""
if len(chat_history) > 0:
    for h in chat_history[-2:]:   # Only last 2 turns
        history_text += f"User: {h['user']}\nAI: {h['ai']}\n"

response = reformulator_chain.invoke({
    "history": history_text,
    "question": query
})
actual_query = response.content.strip()
```

### 🔑 Important Keywords

| Term | Meaning |
|------|---------|
| `chat_history[-2:]` | Python slice — gets only the last 2 items from the list. Keeps token usage low while still handling immediate follow-ups. |
| `response.content.strip()` | `.strip()` removes any leading/trailing whitespace or newlines from the reformulated query before using it. |
| `chat_history.append({...})` | Saves the reformulated query and AI answer to history. The reformulated query is stored (not the original raw input) so future follow-ups make sense. |

---

# 2. Complete Data Flow

## Ingestion + Query Flow

```
═══════════════════════════════════════════════════════════════
                    PHASE 1: INGESTION (One-Time)
═══════════════════════════════════════════════════════════════

  📄 Dsa.pdf
       │
       ▼
  PyPDFLoader
  └── Loads all pages as LangChain Document objects
       │
       ▼
  vocab_save.py :: train_and_save_bm25()
  └── Fits BM25 on raw page text
  └── Saves vocabulary → bm_25_vocab_params.json
       │
       ▼
  RecursiveCharacterTextSplitter
  └── chunk_size=2200, chunk_overlap=150
  └── Creates ~N chunk objects with .page_content + .metadata
       │
       ▼
  For each batch of 10 chunks:
  ┌─────────────────────────────────────────────────────┐
  │  texts = [chunk.page_content, ...]                  │
  │                                                     │
  │  Gemini embeddings_model.embed_documents(texts)     │
  │  └── Returns dense_vectors: list of 768-dim floats  │
  │                                                     │
  │  BM25Encoder.encode_documents(texts)                │
  │  └── Returns sparse_vectors: [{indices, values}]   │
  │                                                     │
  │  Package: {id, values, sparse_values, metadata}     │
  │  └── metadata["text"] = raw chunk text              │
  └─────────────────────────────────────────────────────┘
       │
       ▼
  Pinecone index.upsert()
  └── Each chunk stored as a HYBRID VECTOR in Pinecone
  └── Pinecone Index: "ragproject"

  ✅ INGESTION COMPLETE


═══════════════════════════════════════════════════════════════
               PHASE 2: QUERYING (Every User Interaction)
═══════════════════════════════════════════════════════════════

  👤 User types: "what are its time complexities?"
       │
       ▼
  retrieve.py
  └── Builds history from last 2 turns
  └── Sends to Groq reformulator LLM (temp=0.0)
       │
       ▼
  Reformulated Query:
  "What are the time complexities of a Binary Search Tree?"
       │
       ▼
  rag.py :: get_rag_answer(query_text)
       │
       ├── Gemini: embed_query(query_text)
       │   └── dense_vec: [0.12, -0.45, ...] (768 floats)
       │
       ├── BM25: encode_queries(query_text)
       │   └── sparse_vec: {indices:[42,87,201], values:[0.8,0.6,0.4]}
       │
       ├── Alpha Calculation
       │   └── len(words) > 3 → alpha = 0.7
       │   └── scaled_dense = dense_vec × 0.7
       │   └── scaled_sparse values = sparse_vec × 0.3
       │
       ▼
  Pinecone index.query()
  └── Hybrid search: dense + sparse combined
  └── top_k=3 matches returned with metadata
       │
       ▼
  reranker.py :: rerank_pinecone_matches()
  └── Extracts raw text from 3 Pinecone matches
  └── Sends to Cohere rerank-v3.5
  └── Cohere reorders by true relevance
  └── Reconstructs with page numbers + source info
       │
       ▼
  context_text (3 formatted chunks with page labels)
       │
       ▼
  LangChain Chain: prompt | groq_llm
  └── System prompt from system.txt
  └── context_text injected into {context}
  └── query injected into {question}
  └── Groq (llama-3.1-8b-instant) generates answer
       │
       ▼
  💬 Final Answer displayed to user
       │
       ▼
  chat_history.append({user: query, ai: answer})
```

---

# 3. Concept Q&A

## Core RAG Concepts

---

## ❓ What are Sparse Vectors?

A sparse vector is a vector where **most values are zero** — only a few positions have non-zero values.

**Example:**
Suppose your vocabulary has 10,000 words. For the sentence `"binary tree traversal"`, only 3 words match → only 3 positions are non-zero:

```
Position 0:       0.0
Position 1:       0.0
...
Position 482:     0.87   ← "binary"
...
Position 1203:    0.74   ← "tree"
...
Position 3891:    0.62   ← "traversal"
...
Position 9999:    0.0
```

Only indices `[482, 1203, 3891]` with values `[0.87, 0.74, 0.62]` are stored — the rest are discarded. This is why they're called "sparse" — mostly empty.

**In your code:** `sparse_vec = {'indices': [482, 1203, 3891], 'values': [0.87, 0.74, 0.62]}`

---

## ❓ Sparse vs Dense Vectors — What's the Difference?

| Property | Sparse Vectors (BM25) | Dense Vectors (Gemini) |
|----------|----------------------|----------------------|
| **What they capture** | Exact keywords, specific terms | Meaning, semantics, context |
| **Size** | 10,000+ dimensions, mostly zeros | 768 dimensions, all non-zero |
| **Good at** | Exact term matching ("mutex", "O(log n)") | Conceptual similarity ("how does locking work?") |
| **Bad at** | Synonyms, paraphrasing | Rare technical terms, exact matches |
| **Example win** | Query: "BFS" → finds chunks with "BFS" | Query: "how to visit all nodes" → finds BFS chunks |
| **Example fail** | Query: "graph traversal" may miss "BFS" chunk | Query: "mutex" may return semantically close but wrong chunks |
| **Created by** | BM25 algorithm | Neural network (Gemini) |

**Why use both?** They cover each other's weaknesses. A keyword like `"semaphore"` might not have a close semantic neighbour in embedding space — sparse catches it. A conceptual question like `"why do we need synchronization"` won't match exact keywords — dense catches it.

---

## ❓ How Does BM25 Work Here?

BM25 stands for **Best Match 25**. It's a ranking algorithm that scores documents based on keyword frequency.

**The core idea:**

1. **Term Frequency (TF):** If the query word `"deadlock"` appears 5 times in a chunk, that chunk is more relevant. But the score doesn't grow infinitely — it's capped (saturated).

2. **Inverse Document Frequency (IDF):** If `"deadlock"` appears in only 2 out of 100 chunks, it's more important than `"the"` which appears everywhere. Rare words = higher weight.

3. **Document Length Normalization:** Long chunks naturally contain more words. BM25 normalizes so that a short chunk with 1 mention isn't unfairly beaten by a long chunk with 5 mentions just due to length.

**In your system, BM25 is used in two steps:**

**Step 1 — `.fit()` during `vocab_save.py`:**
BM25 reads all your PDF pages and learns:
- Which words exist in your corpus
- How frequently each word appears across all documents
- This builds the IDF table — word → importance score

**Step 2 — `.encode_documents()` and `.encode_queries()` during ingestion/retrieval:**
- For each chunk/query, BM25 looks up which words are present
- Multiplies TF × IDF for each word
- Returns a sparse vector with scores for matched words only

---

## ❓ Why is Reranking Important?

Vector similarity (both dense and sparse) has a fundamental limitation: **it scores each document independently against the query** without deeply reading both together.

**The problem:**
Pinecone's similarity search gives you top-3 by approximate vector distance. This is fast but imprecise. A chunk might be "close" in vector space but not actually the best answer.

**Reranking solves this:**
Cohere's `rerank-v3.5` is a **cross-encoder** — it reads the query AND the document together at the same time (like a human would) and produces a true relevance score.

```
Without Reranking:           With Reranking:
Pinecone Result 1: Score 0.91 → Actually rank 3 (off-topic terms)
Pinecone Result 2: Score 0.87 → Actually rank 1 (best answer)
Pinecone Result 3: Score 0.85 → Actually rank 2 (good answer)
```

**Real example:**
Query: `"What is the time complexity of mergesort?"`
- Pinecone may return a chunk about `"sorting algorithms overview"` first (high vector similarity due to general sorting terms)
- Reranker will correctly push the chunk that says `"Mergesort: O(n log n) in all cases"` to rank 1

**Cost of reranking:** It's slower and costs API calls. This is why you first use Pinecone to narrow down to 3 candidates fast, then rerank only those 3 — not the entire database.

---

## ❓ What is the Difference Between `encode_documents()` and `encode_queries()`?

Both use the same BM25 vocabulary, but they're meant for different purposes:

| Method | Used for | Applied during |
|--------|----------|----------------|
| `encode_documents()` | The text chunks being stored | Ingestion (`ingest_data.py`) |
| `encode_queries()` | The user's search query | Retrieval (`rag.py`) |

The distinction matters because in BM25, documents and queries are scored asymmetrically — queries are typically short, documents are long. The encoder handles this internally.

---

## ❓ What is Alpha Scaling and Why is it Dynamic?

Alpha (α) controls how much weight to give to dense vs sparse results during hybrid search.

```
Final Score = (α × dense_score) + ((1-α) × sparse_score)
```

**Why dynamic (not fixed)?**
- A query like `"BFS"` (2 words) → you want keyword matching to dominate → α = 0.3 (sparse gets 70%)
- A query like `"explain how deadlock occurs when two processes wait for each other"` (long, conceptual) → you want semantic understanding → α = 0.7 (dense gets 70%)

A fixed alpha would either make short exact queries too vague or long semantic queries too keyword-dependent. Dynamic alpha adapts to the query type automatically.

---

## ❓ Why Store `metadata["text"]` in Pinecone?

Pinecone stores vectors for similarity search — but by default it doesn't store the original text. If you only store the vector, you'd get back a match ID and a score but **no text to give to the LLM**.

By storing `m["text"] = doc.page_content` in metadata, every match from Pinecone also carries the original text string, which is then:
1. Sent to Cohere for reranking
2. Formatted with page numbers and source file names
3. Injected into the Groq LLM prompt as context

Without this, the entire pipeline breaks — the LLM would have nothing to read.

---

## ❓ What is a Cross-Encoder vs Bi-Encoder?

| Type | How it works | Used where |
|------|-------------|-----------|
| **Bi-Encoder** | Encodes query and document separately into vectors, then compares | Gemini + BM25 in Pinecone (fast, scalable) |
| **Cross-Encoder** | Reads query AND document together in one pass | Cohere reranker (slower, more accurate) |

Your system uses both — bi-encoders for fast first-pass retrieval over the entire database, cross-encoder for precise reranking of the final candidates. This is called a **retrieval pipeline** and is the industry standard for production RAG systems.

---

## ❓ Why Use Groq Instead of OpenAI or Gemini?

Groq uses custom hardware (LPUs — Language Processing Units) that runs LLMs significantly faster than standard GPU-based providers. For a conversational Q&A system, low latency matters — you don't want to wait 10 seconds for an answer.

`llama-3.1-8b-instant` is a small but capable open-source model, and Groq's hardware makes it respond almost instantly. It's also free-tier friendly for development.

---

## ❓ Why Does the Reformulator Use `temperature=0.0` but RAG Uses `0.2`?

| Component | Temperature | Reason |
|-----------|------------|--------|
| Query Reformulator (`retrieve.py`) | `0.0` | Must produce a deterministic, clean rewrite. No creativity needed. |
| Answer Generator (`rag.py`) | `0.2` | Slight flexibility for phrasing answers naturally, while staying factual. |

If the reformulator had temperature > 0, the same follow-up question could be rewritten differently each time — which would make the system feel inconsistent and unpredictable.

---

## ❓ What Would Happen if You Skip Reranking?

The top result from Pinecone hybrid search is good, but not perfect. Without reranking:
- The LLM might read the least relevant chunk first and base its answer on it
- Page label / source information is less reliable in terms of ordering
- Answers may be slightly less precise for complex questions

With reranking, the most relevant chunk is always first in the context — the LLM reads the best content first and produces better answers.



---

# 4. Testing & Evaluation — `testing/evaluate.py`

---

## 📄 `testing/evaluate.py`

**Purpose:** Automated offline evaluation of the entire RAG pipeline. Runs 100 questions from a golden dataset, collects answers + context, and scores the pipeline using the Ragas framework with an AI judge.

**File location:** `testing/evaluate.py` — lives in its own subfolder, uses `sys.path.append` to import from the parent project.

---

### Block 1 — Path Fix & Imports

```python
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reranker import rerank_pinecone_matches
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
```

### 🔑 Important Keywords

| Term | Meaning |
|------|---------|
| `sys.path.append(...)` | Adds the parent directory to Python's module search path. Without this, `from reranker import ...` would fail because `evaluate.py` is inside a subfolder. |
| `os.path.dirname(__file__)` | Gets the directory of the current file (`testing/`). |
| `os.path.join(..., '..')` | Goes one level up — back to the project root. |
| `ragas` | The evaluation framework. It uses AI judges (LLMs + embeddings) to score your RAG pipeline across 4 metrics. |
| `faithfulness` | Ragas metric: measures if the answer is grounded in the retrieved context (no hallucination). |
| `answer_relevancy` | Ragas metric: measures if the answer directly addresses the question asked. |
| `context_precision` | Ragas metric: measures if the most relevant chunks are ranked first in the retrieved context. |
| `context_recall` | Ragas metric: measures if all the facts needed to answer the question were found in the retrieved chunks. |

---

### Block 2 — Pipeline Setup (Mirror of `rag.py`)

```python
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("ragproject")

bm25_encoder = BM25Encoder()
bm25_encoder.load("../sparse_vectors/bm_25_vocab_params.json")

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
groq_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, max_retries=2)

# AI Judges (separate instances — used only by Ragas, not by RAG pipeline)
judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
```

**Why two separate LLM instances?**
- `groq_llm` (temp=0.2) is the **generator** — it answers questions with slight flexibility for natural phrasing
- `judge_llm` (temp=0.0) is the **evaluator** — it judges answer quality with zero randomness. You want a deterministic judge, not a creative one

**Why `models/embedding-001` for judge but `gemini-embedding-001` for RAG?**
Ragas's `answer_relevancy` metric uses embeddings to measure semantic similarity between the question and the answer. `embedding-001` is used here as it's the standard Gemini embedding model compatible with Ragas. The RAG pipeline continues to use `gemini-embedding-001` for its hybrid search.

---

### Block 3 — Load the Golden Dataset

```python
df = pd.read_csv("./dataset.csv")
questions = df["question"].tolist()
ground_truths = df["ground_truth"].tolist()
```

### 🔑 Important Keywords

| Term | Meaning |
|------|---------|
| `dataset.csv` | Your **golden dataset** — 100 manually written question + verified answer pairs based strictly on your DSA/OS notes. This is the ground truth. |
| `ground_truth` | The correct, human-verified answer for each question. Ragas compares the AI's answer against this. |
| `pd.read_csv()` | Reads the CSV into a pandas DataFrame. Each row = one question-answer pair. |
| `.tolist()` | Converts the DataFrame column into a plain Python list for easy iteration. |

---

### Block 4 — The Test Loop (Core)

```python
for i, query_text in enumerate(questions):
    time.sleep(10)  # Rate limit buffer

    # A. HYBRID SEARCH (same logic as rag.py)
    dense_vec = embeddings_model.embed_query(query_text)
    sparse_vec = bm25_encoder.encode_queries(query_text)
    alpha = 0.3 if len(words) <= 3 else 0.7
    # ... scaling ...
    query_results = index.query(vector=scaled_dense, sparse_vector=scaled_sparse, top_k=3, include_metadata=True)

    # B. RERANKING
    context_text, context_pieces = rerank_pinecone_matches(
        query=query_text,
        pinecone_matches=query_results['matches'],
        top_n=3
    )
    contexts.append(context_pieces)   # <-- list of strings, not one big string

    # C. GENERATE ANSWER
    response = rag_chain.invoke({"context": context_text, "question": query_text})
    answers.append(response.content)
```

### 🔑 Key Difference from `rag.py`

Notice `rerank_pinecone_matches` now returns **two values**: `context_text, context_pieces`.

- `context_text` → one big formatted string → fed to the LLM for answer generation (same as before)
- `context_pieces` → **list of individual chunk strings** → fed to Ragas separately

Ragas needs `contexts` as a **list of lists** — each question has a list of its retrieved chunk strings. This is different from the LLM which just needs one combined string. The reranker was updated to return both formats to support this.

| Variable | Type | Used by |
|----------|------|---------|
| `context_text` | `str` — one combined string | Groq LLM (answer generation) |
| `context_pieces` | `list[str]` — individual chunks | Ragas (metric calculation) |

---

### Block 5 — Ragas Evaluation

```python
data = {
    "question": questions,       # list of 100 questions
    "answer": answers,           # list of 100 AI-generated answers
    "contexts": contexts,        # list of 100 lists (each = 3 chunks)
    "ground_truth": ground_truths # list of 100 verified correct answers
}
dataset = Dataset.from_dict(data)

result = evaluate(
    dataset=dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    llm=judge_llm,
    embeddings=judge_embeddings,
    max_workers=1,
    raise_exceptions=False
)
```

### 🔑 Important Keywords

| Parameter | Meaning |
|-----------|---------|
| `Dataset.from_dict(data)` | Converts the Python dict into a HuggingFace `Dataset` object — Ragas requires this format. |
| `metrics=[...]` | The 4 metrics Ragas will calculate. Each uses a different evaluation strategy. |
| `llm=judge_llm` | The LLM Ragas uses internally as its AI judge. It reads your answers + context and scores them. |
| `embeddings=judge_embeddings` | The embedding model Ragas uses for `answer_relevancy` — it measures semantic closeness between question and answer. |
| `max_workers=1` | Forces Ragas to run evaluations one at a time (no parallelism). This avoids hitting Groq/Gemini rate limits on free tiers. |
| `raise_exceptions=False` | If one question fails to evaluate (e.g., API timeout), skip it and continue instead of crashing the whole evaluation. |

---

### Block 6 — Results Output

```python
result_df = result.to_pandas()
print(f"Faithfulness:        {result_df['faithfulness'].mean():.2f}")
print(f"Answer Relevancy:    {result_df['answer_relevancy'].mean():.2f}")
print(f"Context Precision:   {result_df['context_precision'].mean():.2f}")
print(f"Context Recall:      {result_df['context_recall'].mean():.2f}")
result_df.to_csv("evaluation_results.csv", index=False)
```

- `.mean()` across 100 rows gives the **overall pipeline score** for each metric (0.0 to 1.0 scale)
- `evaluation_results.csv` saves per-question scores — you can inspect which specific questions failed and why
- `.2f` formats to 2 decimal places: `0.87` not `0.8712345...`

---

## 📊 Evaluation Flow

```
═══════════════════════════════════════════════════════════════
              PHASE 3: EVALUATION (testing/evaluate.py)
═══════════════════════════════════════════════════════════════

  📄 testing/dataset.csv
  └── 100 rows: question | ground_truth
       │
       ▼
  For each of 100 questions:
  ┌─────────────────────────────────────────────────────────┐
  │  time.sleep(10)  ← Rate limit buffer for free APIs      │
  │                                                         │
  │  embed_query()       → dense_vec                        │
  │  encode_queries()    → sparse_vec                       │
  │  alpha calculation   → dynamic weighting                │
  │                                                         │
  │  Pinecone hybrid query (top_k=3)                        │
  │       │                                                 │
  │       ▼                                                 │
  │  rerank_pinecone_matches()                              │
  │  └── Returns context_text (str) + context_pieces (list) │
  │                                                         │
  │  rag_chain.invoke()  → answer (str)                     │
  │                                                         │
  │  answers.append(answer)                                 │
  │  contexts.append(context_pieces)  ← list of 3 chunks   │
  └─────────────────────────────────────────────────────────┘
       │
       ▼  (after all 100 questions)
  Build Ragas Dataset:
  {
    question:     [q1, q2, ... q100]
    answer:       [a1, a2, ... a100]      ← AI generated
    contexts:     [[c1a,c1b,c1c], ...]   ← 3 chunks per question
    ground_truth: [gt1, gt2, ... gt100]  ← your verified answers
  }
       │
       ▼
  Ragas evaluate()
  ┌────────────────────────────────────────────────────────┐
  │  judge_llm (Groq, temp=0.0) reads each Q+A+Context     │
  │                                                        │
  │  faithfulness:       Is answer supported by context?   │
  │  answer_relevancy:   Does answer address the question? │
  │  context_precision:  Are best chunks ranked first?     │
  │  context_recall:     Were all needed facts retrieved?  │
  └────────────────────────────────────────────────────────┘
       │
       ▼
  evaluation_results.csv  ← per-question scores
  Console output          ← averaged scores (0.0 → 1.0)
```

---

---

## Evaluation & Ragas Concepts

---

## ❓ What is Ragas and What Does It Do?

**Ragas** (Retrieval Augmented Generation Assessment) is an evaluation framework specifically built to measure how well a RAG pipeline is working. It doesn't just check if the final answer is correct — it measures the quality of *each stage* of your pipeline separately.

Think of it like a report card with 4 subjects instead of one overall grade:

| Metric | Question it answers | What it measures |
|--------|--------------------|--------------------|
| **Faithfulness** | "Did the AI make things up?" | Whether every claim in the answer can be traced back to the retrieved context. Score: 0 (hallucinating) → 1 (fully grounded) |
| **Answer Relevancy** | "Did the AI actually answer the question?" | Semantic similarity between the question and the answer. A vague or off-topic answer scores low. |
| **Context Precision** | "Did the retriever put the best chunks first?" | Whether the most relevant chunks appear at the top of the retrieved list (position matters for LLMs). |
| **Context Recall** | "Did the retriever find everything needed?" | Whether the retrieved chunks contain all the facts present in the ground truth answer. |

---

## ❓ How Does Ragas Actually Work Internally?

Ragas uses an **AI-judge approach** — it sends prompts to an LLM and an embedding model to score each metric. Here's what happens for each:

**Faithfulness:**
1. Ragas extracts all factual claims from the AI's answer (using `judge_llm`)
2. For each claim, it asks the LLM: "Can this claim be supported by the provided context?"
3. Score = supported claims ÷ total claims

**Answer Relevancy:**
1. Ragas generates several "reverse questions" from the answer (using `judge_llm`) — "what question would this answer be responding to?"
2. Embeds those reverse questions and the original question using `judge_embeddings`
3. Score = average cosine similarity between reverse questions and original question

**Context Precision:**
1. For each retrieved chunk, asks the LLM: "Is this chunk relevant to answering the question?"
2. Calculates a weighted precision score that rewards relevant chunks appearing earlier in the list

**Context Recall:**
1. Extracts all factual claims from the `ground_truth` answer
2. For each claim, asks: "Is this claim supported by any retrieved chunk?"
3. Score = claims found in context ÷ total claims in ground truth

---

## ❓ What is a Golden Dataset and Why Must it be "Golden"?

A **golden dataset** is a manually curated set of question + verified answer pairs that represents the ground truth for your system. It's called "golden" because it's the standard everything else is measured against.

**Why it must be manually written:**
- Generated by a human who has actually read the notes
- Each answer is verified to be factually correct based only on what's in the PDF
- Questions cover a range of types: factual, conceptual, comparison, application
- It cannot be generated by the same AI being evaluated — that would be circular

**In your project:** `dataset.csv` has 100 questions + answers written from your DSA and OS notes. Every Ragas score is calculated relative to this dataset — if your ground truth is wrong or vague, your scores will be meaningless even if the AI performs perfectly.

---

## ❓ What Does `time.sleep(10)` Do in the Test Loop and Why is it Necessary?

```python
time.sleep(10)  # Speed bump for free API tiers
```

This pauses execution for 10 seconds between each question. It's a rate limit buffer.

**Why it's needed:**
Free API tiers (Groq, Gemini) have **requests-per-minute (RPM)** limits. Running 100 questions back-to-back would exhaust the rate limit in under a minute, causing API errors and failed evaluations. By sleeping 10 seconds between each question, the script runs at 6 questions/minute — well within free-tier limits.

**Trade-off:** 100 questions × 10 seconds = ~17 minutes to complete one full evaluation run. This is why automated CI/CD evaluation is typically run only on push events, not on every local change.

---

## ❓ Why Does `contexts` Need to be a List of Lists?

```python
contexts.append(context_pieces)   # list of strings, NOT one big string
```

Ragas requires `contexts` to be `list[list[str]]` — a list where each element is a list of the individual retrieved chunk strings for that question.

```
contexts = [
    ["chunk text A", "chunk text B", "chunk text C"],   ← question 1
    ["chunk text D", "chunk text E", "chunk text F"],   ← question 2
    ...
]
```

This is because Ragas evaluates **each chunk independently** for metrics like `context_precision` and `context_recall`. If you passed one big merged string, Ragas couldn't tell where one chunk ends and another begins — it would treat the entire context as a single document and scoring would be wrong.

---

## ❓ What is `raise_exceptions=False` and When Would You Change it?

```python
result = evaluate(..., raise_exceptions=False)
```

With `raise_exceptions=False`, if Ragas fails to evaluate a single question (API timeout, malformed response, rate limit), it marks that row as `NaN` and continues to the next question. The final `.mean()` simply ignores `NaN` rows.

**When to use `True`:** In a CI/CD pipeline where you want the build to fail hard if any evaluation breaks — useful for debugging new issues. In development with free-tier APIs, `False` is safer because transient failures won't abort a 17-minute evaluation run.

---

## ❓ What is the Difference Between Context Precision and Context Recall?

These two metrics measure opposite directions of retrieval quality:

| | Context Precision | Context Recall |
|--|------------------|----------------|
| **Question** | "Of what was retrieved, how much was useful?" | "Of what was needed, how much was retrieved?" |
| **Analogy** | Search results — were the top results relevant? | Search completeness — did you find all the answers? |
| **Low score means** | Retriever is returning noisy, irrelevant chunks | Retriever is missing important chunks that contain needed facts |
| **Fixed by** | Better reranking, tighter retrieval (lower top_k) | More chunks (higher top_k), better embeddings, hybrid search |

In your system, a **low context precision** suggests reranking isn't working well. A **low context recall** suggests the hybrid search isn't finding all relevant content — possibly needs better chunking or a higher `top_k`.

---

## ❓ Why is Automated Evaluation Important for a RAG System?

Without evaluation, you're flying blind. Here's what can go wrong silently:

1. **Prompt changes break things** — You tweak `system.txt` to improve one type of answer, and it accidentally makes 20 other questions worse. Without evaluation, you'd never know.
2. **Embedding model updates** — If Google updates `gemini-embedding-001`, your stored vectors and new query vectors might drift. Scores would drop silently.
3. **Retrieval regressions** — A change to chunking strategy that seems reasonable might hurt context recall. Numbers catch this; intuition doesn't.

**The eval → fix → eval loop** is what separates a project from a production system:
```
Change something
     ↓
Run evaluate.py
     ↓
Check 4 scores
     ↓
If any score dropped → investigate that metric's CSV rows → fix it
```

---

## ❓ Summary of Models Used

| Model | Provider | Used for | Where |
|-------|----------|---------|-------|
| `gemini-embedding-001` | Google Gemini | Dense embeddings (768-dim) | `ingest_data.py`, `rag.py`, `evaluate.py` |
| `embedding-001` | Google Gemini | Ragas judge embeddings (answer relevancy) | `evaluate.py` only |
| `BM25Encoder` | Pinecone Text | Sparse embeddings | `vocab_save.py`, `ingest_data.py`, `rag.py`, `evaluate.py` |
| `llama-3.1-8b-instant` (temp=0.2) | Groq | Answer generation | `rag.py`, `evaluate.py` |
| `llama-3.1-8b-instant` (temp=0.0) | Groq | Query reformulation + Ragas AI judge | `retrieve.py`, `evaluate.py` |
| `rerank-v3.5` | Cohere | Reranking top-k results | `reranker.py`, `evaluate.py` |