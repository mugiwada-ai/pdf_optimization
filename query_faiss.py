import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def run_temple_chatbot_search(user_query, top_k=1):
    print(f"User Query: '{user_query}'\n")
    
    print("1. Loading Database and Assets...")
    # Load the mathematical FAISS index
    index = faiss.read_index("temple_vectors.faiss")
    
    # Load the mapping metadata (Note: JSON saves integer keys as strings)
    with open("temple_metadata.json", "r", encoding="utf-8") as f:
        metadata_map = json.load(f)
        
    # Load the massive Parent Chunks
    with open("llm_parents.json", "r", encoding="utf-8") as f:
        llm_parents = json.load(f)

    # Load the embedding model (it will load instantly from your local cache)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    print("2. Searching FAISS...")
    # Embed the user's question into a mathematical vector
    query_vector = embedder.encode([user_query]).astype('float32')
    
    # Perform the search. Returns the distances (scores) and the FAISS IDs
    distances, indices = index.search(query_vector, top_k)
    
    print("\n--- SEARCH RESULTS ---")
    for i in range(top_k):
        # 1. Get the mathematical FAISS ID
        faiss_id = str(indices[0][i])
        distance = distances[0][i]
        
        # 2. Look up the metadata using the FAISS ID
        chunk_metadata = metadata_map[faiss_id]
        parent_id = chunk_metadata["parent_id"]
        
        # 3. Retrieve the massive Parent Context using the parent_id
        parent_context = llm_parents[parent_id]
        
        print(f"Match #{i+1} (Score: {distance:.4f})")
        print(f"Found in Section: {chunk_metadata['h1']} -> {chunk_metadata['h2']}")
        print(f"Child Metadata Parent ID: {parent_id}")
        print("\n[CONTEXT DELIVERED TO LLM]:")
        print(parent_context["full_text"])
        print("-" * 50)

if __name__ == "__main__":
    # Let's ask a question based on the "Background" chunks you showed me earlier!
    test_question = "what are donor priviliges?"
    run_temple_chatbot_search(test_question)