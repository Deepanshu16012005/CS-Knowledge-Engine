import os
import cohere
from dotenv import load_dotenv

# Load env variables and initialize Cohere
load_dotenv()
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

def rerank_pinecone_matches(query: str, pinecone_matches: list, top_n: int = 3) -> str:
    """
    Takes raw Pinecone matches, reranks them using Cohere, 
    and returns a formatted context string of the top N results.
    """
    # 1. Safety check: Did Pinecone find anything?
    if not pinecone_matches:
        return ""

    # 2. Extract just the text strings for Cohere to read
    docs_to_rerank = [match['metadata'].get('text', '') for match in pinecone_matches]

    # 3. Call Cohere to rerank the text
    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=docs_to_rerank,
        top_n=top_n
    )

    # 4. Reconstruct the context with Metadata (Page numbers, Sources)
    final_context_pieces = []
    
    for r in rerank_response.results:
        # r.index tells us which original Pinecone dictionary to pull from
        original_match = pinecone_matches[r.index]
        meta = original_match['metadata']
        
        content = meta.get('text', "")
        page = meta.get('page_label', 'Unknown')
        source = os.path.basename(meta.get('source', 'unknown'))
        
        # Format for Groq
        formatted_chunk = f"\n--- (File: {source} | Page: {page}) ---\n{content}"
        final_context_pieces.append(formatted_chunk)

    # Combine the top 3 chunks into one big string
    return "\n\n".join(final_context_pieces) , final_context_pieces