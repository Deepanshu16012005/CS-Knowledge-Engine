import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate   
load_dotenv()
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = PineconeVectorStore(
    index_name="deepanshu", 
    embedding=embeddings_model
)

groq_llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.2, 
    max_retries=2
)
system_instruction = """You are a technical assistant specialized in Data Structures & Algorithms (DSA) and Operating Systems (OS).
Your task is to answer the user's question ONLY using the provided context.

STRICT RULES:
1. Use ONLY the information present in the provided context.
2. Do NOT guess or hallucinate information.
3. If the answer is not present, respond with: "The answer is not available in the provided context."
4. Prefer bullet points or step-by-step explanations for clarity.

Context from notebook:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("user", "{question}")
])

rag_chain = prompt | groq_llm

def get_rag_answer(query_text: str) -> str:
    """Takes a formulated query, searches Pinecone, and returns the AI answer."""
    print("Searching database...")
    
    # Fetch the top 3 most relevant chunks
    results = vectorstore.similarity_search(query_text, k=3)
    context_text = "\n\n".join([doc.page_content for doc in results])
    
    print("Generating answer using Groq...")
    # Invoke the chain with the retrieved context and the user's question
    response = rag_chain.invoke({
        "context": context_text, 
        "question": query_text
    })
    
    return response.content

print("✅ RAG Engine Loaded (Groq Generation) & Connected to Pinecone Database!")