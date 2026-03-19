# 📚 CS Knowledge Base — RAG Q&A System

A Retrieval-Augmented Generation (RAG) pipeline built to answer questions on **Data Structures & Algorithms (DSA)** and **Operating Systems (OS)** from your own PDF notes.

It uses **Google Gemini** for embeddings, **Pinecone** as the vector database, and **Groq (LLaMA 3.1)** for fast answer generation — with multi-turn **chat history** support.

---

## 📁 Project Structure

```
CS-KNOWLEDGEBASE/
├── pdf/                  # Your source PDF notes go here
├── rag_env/              # Python virtual environment (local only)
├── .env                  # Your actual API keys (never commit this)
├── .env.example          # Template for environment variables
├── .gitignore            # Ignores .env, rag_env, __pycache__, etc.
├── ingest_data.py        # Load, chunk & upload PDF to Pinecone
├── rag.py                # Retrieval from Pinecone + answer generation via Groq
├── retrieve.py           # Chat loop with query reformulation
└── requirements.txt      # Python dependencies
```

---

## ⚙️ How It Works

```
📄 PDF Notes
     │
     ▼
ingest_data.py
  ├── Loads PDF using PyPDFLoader
  ├── Splits into chunks (size: 2200, overlap: 150)
  ├── Embeds chunks via Google Gemini (gemini-embedding-001)
  └── Uploads to Pinecone in batches of 10

     │
     ▼ (One-time setup done ✅)

User asks a question
     │
     ▼
retrieve.py
  ├── Maintains chat history (last 2 turns)
  ├── Reformulates vague/follow-up queries into standalone questions
  └── Calls get_rag_answer() from rag.py

     │
     ▼
rag.py
  ├── Embeds the reformulated query via Gemini
  ├── Fetches top 3 relevant chunks from Pinecone
  ├── Builds a prompt with retrieved context
  └── Generates a grounded answer using Groq (llama-3.1-8b-instant)

     │
     ▼
💬 Final Answer
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **LangChain** | Orchestration framework |
| **Google Gemini** (`gemini-embedding-001`) | Generating text embeddings |
| **Pinecone** | Vector database for storing & retrieving chunks |
| **Groq** (`llama-3.1-8b-instant`) | Fast LLM inference for answer generation & query reformulation |
| **PyPDFLoader** | Loading PDF documents |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd CS-KNOWLEDGEBASE
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv rag_env

# On Windows
rag_env\Scripts\activate

# On Mac/Linux
source rag_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

> Get your keys from:
> - Gemini: https://aistudio.google.com/app/apikey
> - Groq: https://console.groq.com/keys
> - Pinecone: https://app.pinecone.io/

### 5. Add Your PDF

Place your PDF file inside the `pdf/` folder and update the path in `ingest_data.py`:

```python
pdf_path = "pdf/your-file.pdf"   # line 4 in ingest_data.py
```

### 6. Ingest Data into Pinecone *(Run Once)*

```bash
python ingest_data.py
```

This will chunk your PDF, embed it using Gemini, and upload everything to Pinecone in batches. It pauses 10 seconds between batches to respect Gemini's rate limits.

### 7. Start Asking Questions

```bash
python retrieve.py
```

---

## 💬 Chat Features

- **Multi-turn conversation** — the system remembers your last 2 interactions
- **Query reformulation** — vague follow-ups like *"explain it more"* are automatically rewritten into complete standalone questions before searching
- Type `clear` to reset chat history
- Type `exit` to quit

**Example session:**
```
What would you like to search for? What is a Binary Search Tree?
Formulated Query: What is a Binary Search Tree?
Answer: A Binary Search Tree (BST) is a node-based data structure where...

What would you like to search for? What are its time complexities?
Formulated Query: What are the time complexities of a Binary Search Tree?
Answer: The time complexities for BST operations are...
```

---

## ⚠️ Notes

- The system answers **only** from your uploaded PDF context — it will not hallucinate or use outside knowledge.
- If the answer isn't in the PDF, it will respond: *"The answer is not available in the provided context."*
- The Pinecone index name is currently hardcoded as `"deepanshu"` in both `ingest_data.py` and `rag.py` — update this to your own index name.

---

## 📄 License

This project is open-source. Feel free to use and modify.
