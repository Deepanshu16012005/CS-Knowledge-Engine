from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
# 1. Define the path to your PDF file
pdf_path = ".pdf"  # Replace with your actual file path

# 2. Initialize the loader
loader = PyPDFLoader(pdf_path)

# 3. Load the document
documents = loader.load()

# 4. Check the output
print(f"Successfully loaded {len(documents)} pages.")

# 5. splitting the document
text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=2200,
   chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)

# 6. embedding function 

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 2. Initialize the Gemini Embedding model
# Updated to Google's current active embedding model!
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 3. Test it out
test_embedding = embeddings_model.embed_query("This is a test document.")
print(f"\nSuccessfully generated an embedding with {len(test_embedding)} dimensions using Gemini!")

from langchain_pinecone import PineconeVectorStore


# 2. Define your index name (must match exactly what you named it on the website)
index_name = "deepanshu" 

print("\nUploading chunks to Pinecone... this might take a moment.")

# 3. Upload the chunks to your cloud database!
# This takes your 'chunks' from Step 2 and your 'embeddings_model' from Step 3
import time

# 1. Initialize the connection to your Pinecone index
vectorstore = PineconeVectorStore(
    index_name=index_name, 
    embedding=embeddings_model
)

print(f"\nUploading {len(chunks)} chunks in batches to avoid rate limits...")

# 2. Upload in batches of 10 with a pause
batch_size = 10

for i in range(0, len(chunks), batch_size):
    # Grab a slice of 10 chunks
    batch = chunks[i : i + batch_size]
    
    # Upload just this small batch
    vectorstore.add_documents(batch)
    print(f"Uploaded chunks {i} to {i + len(batch)}...")
    
    # Pause for 10 seconds to let Gemini's rate limit cool down
    time.sleep(10) 

print("Successfully uploaded ALL document embeddings to Pinecone Cloud!")

# 3. Test the Retrieval
query = "What is the main topic of this document?" 

print(f"\nSearching Pinecone for: '{query}'")
results = vectorstore.similarity_search(query, k=2)

print("\n--- Top Match Found in Pinecone ---")
print(results[0].page_content)
# 4. Test the Retrieval from the Cloud!
query = "What is the main topic of this document?" 

print(f"\nSearching Pinecone for: '{query}'")
results = vectorstore.similarity_search(query, k=2)

print("\n--- Top Match Found in Pinecone ---")
print(results[0].page_content)