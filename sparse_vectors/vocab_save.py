from pinecone_text.sparse import BM25Encoder
import json

def train_and_save_bm25(documents_list, save_path="bm25_params.json"):
    """
    Trains the BM25 encoder on your specific notes and saves the vocabulary.
    
    Args:
        documents_list (list of str): A list containing all the text chunks from your PDFs/notes.
        save_path (str): Where to save the trained model.
    """
    print("Initializing BM25 Encoder...")
    bm25 = BM25Encoder()
    
    print(f"Training on {len(documents_list)} chunks. This might take a moment...")
    # This step is crucial: it learns that words like "mutex" and "sliding" 
    # are highly important in your specific dataset.
    bm25.fit(documents_list)
    
    print(f"Saving trained vocabulary to {save_path}...")
    bm25.dump(save_path)
    print("✅ BM25 Encoder successfully trained and saved!")

