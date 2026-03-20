import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 

import os
from pinecone import Pinecone  # Native client
from pinecone_text.sparse import BM25Encoder

from reranker import rerank_pinecone_matches
load_dotenv()

# 1. Setup the Native Pinecone Client (Required for Hybrid)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("ragproject")

# 2. Load the BM25 Encoder (The same one used in ingestion)
bm25_encoder = BM25Encoder()
bm25_encoder.load("./sparse_vectors/bm_25_vocab_params.json")


embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

groq_llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.2, 
    max_retries=2
)
with open("./prompts/system.txt", "r") as file:
    system_instruction = file.read()

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("user", "{question}")
])

rag_chain = prompt | groq_llm

def get_rag_answer(query_text: str) -> str:
    """Takes a formulated query, searches Pinecone, and returns the AI answer."""
    
    
    # --- STEP A: GENERATE BOTH VECTORS ---
    # Dense Vector (Meaning)
    dense_vec = embeddings_model.embed_query(query_text)
    # Sparse Vector (Keywords)
    sparse_vec = bm25_encoder.encode_queries(query_text)

    words = query_text.split()
    if len(words) <= 3:
        alpha = 0.3  # Short query? Trust keywords more.
    else:
        alpha = 0.7  # Long query? Trust meaning more.
    
    # Multiply every number in the dense list by 0.7
    scaled_dense = [v * alpha for v in dense_vec]
    
    # Multiply every value in the sparse dictionary by (1 - 0.7) = 0.3
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
    print("Got Both Sparse and Dense Vectors From Database\n")

    print("Reranking using COHERE")

    # Ranking top 3 queries
    try:
        context_text, context_pieces = rerank_pinecone_matches(
            query=query_text, 
            pinecone_matches=query_results['matches'], 
            top_n=3
        )
        print("Ranking successful")
    except Exception as e:
        print(f"Ranking Failed ERROR:{e}")
        context_pieces = []
        for match in query_results['matches']:
            page_num = match.metadata.get('page_label', 'Unknown')
            # We explicitly label the text snippet with its page number FOR Groq to read
            formatted_chunk = f"--- (Page: {page_num} \n Source pdf : {match.metadata.get('source', 'unknown')}) ---\n{match.metadata.get('text','no text found under this chunk')}"
            context_pieces.append(formatted_chunk)
        context_text = "\n\n".join(context_pieces)
    
    print("Generating answer using AI...")
    # Invoke the chain with the retrieved context and the user's question
    response = rag_chain.invoke({
        "context": context_text, 
        "question": query_text
    })
    
    return response.content

print("✅ RAG Engine Loaded & Connected to Database!\n")