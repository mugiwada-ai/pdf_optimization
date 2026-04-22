import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_and_save_faiss():
    print("1. Loading Child Chunks...")
    with open("faiss_children.json", "r", encoding="utf-8") as f:
        children = json.load(f)

    # Extract just the text so the model can embed it
    texts_to_embed = [child["search_text"] for child in children]
    
    print(f"2. Generating Vectors for {len(texts_to_embed)} chunks...")
    print("   (This might take 10-30 seconds depending on your CPU)")
    
    # Load the free, local embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate the embeddings (Outputs a numpy array)
    embeddings = embedder.encode(texts_to_embed)
    
    # Convert embeddings to float32 (FAISS strict mathematical requirement)
    embeddings = np.array(embeddings).astype('float32')

    print("3. Building the FAISS Database...")
    # Get the dimension size from the embedding model (MiniLM is 384 dimensions)
    dimension = embeddings.shape[1] 
    
    # Create the FAISS Index (IndexFlatL2 is a standard Euclidean distance search engine)
    index = faiss.IndexFlatL2(dimension)
    
    # Add the vectors into FAISS. 
    # FAISS will automatically assign them integer IDs: 0, 1, 2, 3...
    index.add(embeddings)

    print("4. Saving FAISS Index and Metadata to Disk...")
    # Save the mathematical FAISS index to your hard drive
    faiss.write_index(index, "temple_vectors.faiss")
    
    # FAISS doesn't store metadata natively. We must save a mapping list so we 
    # know that FAISS ID 0 corresponds to children[0].
    metadata_map = {}
    for i, child in enumerate(children):
        metadata_map[i] = child["metadata"]
        
    # Save the mapping dictionary as a JSON file
    with open("temple_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_map, f, indent=4)

    print(f"[Success] Successfully embedded {index.ntotal} vectors!")
    print("[Success] FAISS database built and saved to 'temple_vectors.faiss'")
    print("[Success] Metadata saved to 'temple_metadata.json'")

if __name__ == "__main__":
    build_and_save_faiss()