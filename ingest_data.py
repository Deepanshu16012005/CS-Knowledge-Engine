from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
import os
from pinecone_text.sparse import BM25Encoder
from sparse_vectors.vocab_save import train_and_save_bm25
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()
# 1. Define the path to your PDF file
pdf_path = "./pdf/Dsa.pdf"  # Replace with your actual file path
api_key = os.getenv("PINECONE_API_KEY")

# 2. Initialize the Pinecone client (This defines 'pc')
pc = Pinecone(api_key=api_key)
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




# sparse vectors
text_chunks= [doc.page_content for doc in documents]
#used for making json file 
train_and_save_bm25(text_chunks,"./sparse_vectors/bm_25_vocab_params.json")



#loading the file
bm25_encoder = BM25Encoder()
bm25_encoder.load("./sparse_vectors/bm_25_vocab_params.json")
print("✅ BM25 Vocabulary Loaded!")



# 2. Initialize the Gemini Embedding model
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# # 2. Define your index name (must match exactly what you named it on the website)
index_name = "ragproject"
index = pc.Index(index_name)

print("\nUploading chunks to Pinecone... this might take a moment.")

# # 3. Upload the chunks to your cloud database!

batch_size = 10
for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    
    # Extract the raw text and metadata from the batch
    texts = [doc.page_content for doc in batch]
    metadatas = []
    for doc in batch:
    # 1. Start with the existing metadata (page_label, source, etc.)
        m = doc.metadata.copy() 
    
    # 2. Add the 'text' field so it gets stored in Pinecone
        m["text"] = doc.page_content 
    
        metadatas.append(m)
    
    # Generate unique IDs for this batch (e.g., chunk_0, chunk_1...)
    ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
    # --- THIS IS THE HYBRID MAGIC ---
    # Generate ALL dense vectors for the batch using Gemini
    dense_vectors = embeddings_model.embed_documents(texts)
    
    # Generate ALL sparse vectors for the batch using your BM25 file
    sparse_vectors = bm25_encoder.encode_documents(texts)
    
    # Package them together into Pinecone's required format
    vectors_to_upload = []
    for j in range(len(batch)):
        vectors_to_upload.append({
            "id": ids[j],
            "values": dense_vectors[j],         # Gemini embedding
            "sparse_values": sparse_vectors[j], # BM25 embedding
            "metadata": metadatas[j]
        })
    
    # 4. Upsert the hybrid batch directly into Pinecone
    index.upsert(vectors=vectors_to_upload)
    
    print(f"✅ Uploaded chunks {i} to {i + len(batch) - 1}...")
    time.sleep(10) # Gemini cool-down
print("\n🎉 Ingestion Complete!")