import json
import uuid

def format_parent_child_chunks():
    print("1. Loading the 598 semantic chunks...")
    with open("final_semantic_chunks.json", "r", encoding="utf-8") as f:
        semantic_chunks = json.load(f)

    parents = {}        # Holds the massive Parent Contexts for the LLM
    faiss_children = [] # Holds the small search chunks for FAISS

    current_parent_key = None
    current_parent_id = None
    current_parent_text = ""

    print("2. Grouping chunks by Headings (H1 + H2)...")
    for chunk in semantic_chunks:
        h1 = chunk['h1_context'] or "Unknown_H1"
        h2 = chunk['h2_context'] or "No_H2"
        
        # Create a unique key for the chapter (e.g., "Background_No_H2")
        parent_key = f"{h1}_{h2}"
        
        # If the topic changed (e.g., moving from "Background" to "VIP Darshan")
        if parent_key != current_parent_key:
            # Save the previous Parent Chunk before starting a new one
            if current_parent_id:
                parents[current_parent_id] = {
                    "h1_context": current_parent_key.split('_')[0],
                    "h2_context": current_parent_key.split('_')[1],
                    "full_text": current_parent_text.strip()
                }
                
            # Start tracking the new Parent
            current_parent_key = parent_key
            # Generate a unique ID (e.g., 550e8400-e29b-41d4-a716-446655440000)
            current_parent_id = str(uuid.uuid4()) 
            current_parent_text = ""
        
        # Append the child text to the massive parent block
        current_parent_text += chunk['text'] + " "
        
        # Format the Child Chunk for FAISS Ingestion
        faiss_children.append({
            "search_text": chunk['text'],  # FAISS will only embed this specific sentence/paragraph
            "metadata": {
                "h1": h1,
                "h2": h2,
                "parent_id": current_parent_id # Links the search result back to the massive LLM block
            }
        })

    # Save the very last Parent Chunk after the loop ends
    if current_parent_id:
        parents[current_parent_id] = {
            "h1_context": current_parent_key.split('_')[0],
            "h2_context": current_parent_key.split('_')[1],
            "full_text": current_parent_text.strip()
        }

    print("3. Saving structured files to disk...")
    # Save both to disk
    with open("faiss_children.json", "w", encoding="utf-8") as f:
        json.dump(faiss_children, f, indent=4)

    with open("llm_parents.json", "w", encoding="utf-8") as f:
        json.dump(parents, f, indent=4)

    print(f"[Success] Generated {len(faiss_children)} Child search chunks.")
    print(f"[Success] Generated {len(parents)} massive Parent LLM chunks.")

if __name__ == "__main__":
    format_parent_child_chunks()