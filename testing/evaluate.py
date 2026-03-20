import os
import time
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
import sys
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Your RAG imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder


# Ragas imports
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
from reranker import rerank_pinecone_matches


load_dotenv("../.env")

# ==========================================
# 1. SETUP THE PIPELINE (Copied from your rag.py)
# ==========================================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("ragproject")

bm25_encoder = BM25Encoder()
bm25_encoder.load("../sparse_vectors/bm_25_vocab_params.json")

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
groq_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, max_retries=2)

with open("../prompts/system.txt", "r") as file:
    system_instruction = file.read()

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("user", "{question}")
])
rag_chain = prompt | groq_llm

# Setup AI Judges
judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# ==========================================
# 2. LOAD DATASET
# ==========================================
print("📊 Loading Golden Dataset...")
# Testing 10 questions for now to be safe
df = pd.read_csv("./dataset.csv")

questions = df["question"].tolist()
ground_truths = df["ground_truth"].tolist()

answers = []
contexts = []

# ==========================================
# 3. RUN THE TEST LOOP
# ==========================================
print(f"🤖 Testing your RAG pipeline standalone...")

for i, query_text in enumerate(questions):
    print(f"\n[{i+1}/{len(questions)}] Asking: {query_text[:50]}...")
    time.sleep(10) # Speed bump for free API tiers
    
    # --- A. HYBRID SEARCH ---
    dense_vec = embeddings_model.embed_query(query_text)
    sparse_vec = bm25_encoder.encode_queries(query_text)

    words = query_text.split()
    alpha = 0.3 if len(words) <= 3 else 0.7
    
    scaled_dense = [v * alpha for v in dense_vec]
    scaled_sparse = {
        'indices': sparse_vec['indices'],
        'values': [v * (1 - alpha) for v in sparse_vec['values']]
    }

    query_results = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=3,
        include_metadata=True
    )
    

    # --- B. RERANKING ---
    try:
        context_text , context_pieces = rerank_pinecone_matches(
            query=query_text, 
            pinecone_matches=query_results['matches'], 
            top_n=3
        )
        contexts.append(context_pieces)
    except Exception:
        context_pieces = [f"{m.metadata.get('text','')}" for m in query_results['matches']]
        context_text = "\n\n".join(context_pieces)
        contexts.append(context_pieces)
    # --- C. GENERATE ANSWER ---
    response = rag_chain.invoke({
        "context": context_text, 
        "question": query_text
    })
    
    answers.append(response.content)

# ==========================================
# 4. GRADE THE RESULTS WITH RAGAS
# ==========================================
print("\n⚖️ Handing the results over to the AI Judge...")
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data)

result = evaluate(
    dataset = dataset, 
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    llm=judge_llm,
    embeddings=judge_embeddings,
    max_workers=1,
    raise_exceptions=False
)

print("\n" + "="*50)
print("🏆 FINAL EXAM RESULTS 🏆")
print("="*50)
result_df = result.to_pandas()
print(f"Faithfulness (No Hallucinations): {result_df['faithfulness'].mean():.2f}")
print(f"Answer Relevancy (Direct Answers): {result_df['answer_relevancy'].mean():.2f}")
print(f"Context Precision (Best Chunks First): {result_df['context_precision'].mean():.2f}")
print(f"Context Recall (Found All Facts): {result_df['context_recall'].mean():.2f}")

result_df.to_csv("evaluation_results.csv", index=False)