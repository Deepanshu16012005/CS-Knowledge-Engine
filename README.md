# 🧠 DSA & OS Intelligent Assistant

A high-performance, terminal-based AI assistant built to provide precise answers to complex Computer Science questions using **Retrieval-Augmented Generation (RAG)**. 

This project transforms static study materials and academic notebooks into a searchable, conversational knowledge base, specifically optimized for **Data Structures & Algorithms (DSA)** and **Operating Systems (OS)**.

---

## 🚀 The Tech Stack

* **Orchestration:** [LangChain](https://python.langchain.com/) (LCEL)
* **Inference Engine:** [Groq](https://groq.com/) (Llama-3.1-8B-Instant)
* **Vector Database:** [Pinecone](https://www.pinecone.io/)
* **Embedding Model:** Google Generative AI (`models/gemini-embedding-001`)
* **Security:** `python-dotenv`

---

## ✨ Key Features

* **Zero Hallucinations:** Answers *only* using your provided context.
* **Contextual Memory:** Automatically rewrites follow-up questions (e.g., "Explain its complexity") based on chat history.
* **High Throughput:** Optimized to utilize Groq's 14,400 free requests per day for ultra-low latency.

---

## 🛠️ Installation & Setup

### 1. Clone & Setup
```bash
git clone 
cd 
python -m venv rag_env
source rag_env/bin/activate  # Windows: rag_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```Create a .env file in the root directory and add your keys:
GOOGLE_API_KEY="your_google_key"
PINECONE_API_KEY="your_pinecone_key"
GROQ_API_KEY="your_groq_key"
```

💻 Usage
```To start the interactive chat session:
python retrive.py
```
Type clear to wipe session memory.
Type exit to close.
