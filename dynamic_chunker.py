import json
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the checkpoint data
with open("structured_spans.json", "r", encoding="utf-8") as f:
    tagged_spans = json.load(f)

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def phase1_build_document_tree(tagged_spans):
    """
    Phase 1: Sequential Grouping.
    Converts a flat list of tagged spans into a structured Document Tree.
    """
    document_tree = []
    current_h1 = None
    current_h2 = None
    current_body_text = ""

    def save_current_section():
        """Helper to save the accumulated body text into the tree before moving on."""
        nonlocal current_body_text
        if current_body_text.strip() and (current_h1 or current_h2):
            document_tree.append({
                "h1_context": current_h1,
                "h2_context": current_h2,
                "body_text": current_body_text.strip()
            })
        current_body_text = ""

    for span in tagged_spans:
        tag = span['tag']
        text = span['text']

        if tag == "H1":
            save_current_section()  # Save whatever we were working on
            current_h1 = text
            current_h2 = None       # Reset H2 because we entered a new H1 chapter
            
        elif tag == "H2":
            save_current_section()
            current_h2 = text
            
        elif tag == "Body":
            # Accumulate body text under the current headers
            current_body_text += text + " "

    # Save the very last section after the loop ends
    save_current_section()
    
    return document_tree


def phase2_semantic_splitter(document_tree, similarity_threshold=0.4):
    """
    Phase 2: Semantic Splitter.
    Embeds sentences and mathematically splits them when the topic changes.
    """
    final_semantic_chunks = []

    for section in document_tree:
        body_text = section['body_text']
        
        # 1. Tokenize into individual sentences
        sentences = sent_tokenize(body_text)
        
        # If the section is very short (1-2 sentences), just keep it as one chunk
        if len(sentences) <= 2:
            final_semantic_chunks.append({
                "h1_context": section["h1_context"],
                "h2_context": section["h2_context"],
                "text": body_text
            })
            continue

        # 2. Embed the sentences to convert meaning into vector coordinates
        embeddings = embedder.encode(sentences)
        
        # 3. Calculate cosine similarity between consecutive sentences
        # How similar is sentence[0] to sentence[1]? sentence[1] to sentence[2]?
        similarities = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
        
        # 4. Find the Breakpoints based on the math
        # If the similarity between two sentences drops below the threshold, break the chunk.
        current_chunk_sentences = [sentences[0]]
        
        for i, sim in enumerate(similarities):
            # i corresponds to the similarity between sentence[i] and sentence[i+1]
            if sim < similarity_threshold:
                # The topic changed drastically! Save the current chunk.
                final_semantic_chunks.append({
                    "h1_context": section["h1_context"],
                    "h2_context": section["h2_context"],
                    "text": " ".join(current_chunk_sentences)
                })
                # Start a new chunk with the next sentence
                current_chunk_sentences = [sentences[i+1]]
            else:
                # The topic is still the same, add the next sentence to the chunk
                current_chunk_sentences.append(sentences[i+1])
                
        # Append whatever sentences are left over at the end
        if current_chunk_sentences:
            final_semantic_chunks.append({
                "h1_context": section["h1_context"],
                "h2_context": section["h2_context"],
                "text": " ".join(current_chunk_sentences)
            })

    return final_semantic_chunks
# 2. Run Phase 1: Build the Document Tree
doc_tree = phase1_build_document_tree(tagged_spans)

# 3. Run Phase 2: Semantically Split Oversized Text
final_chunks = phase2_semantic_splitter(doc_tree, similarity_threshold=0.45)

# 4. Save the final structured chunks
with open("final_semantic_chunks.json", "w", encoding="utf-8") as f:
    json.dump(final_chunks, f, indent=4)

print(f"[Success] Saved {len(final_chunks)} semantically split chunks to final_semantic_chunks.json")