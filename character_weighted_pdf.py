import fitz  # PyMuPDF
from collections import defaultdict
import json


def find_baseline_font_size(doc):
    """Step 1: Find the character-weighted baseline font size."""
    font_char_counts = defaultdict(int)
    
    for page in doc:
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        
                        font_size = round(span.get("size", 0.0), 1)
                        font_char_counts[font_size] += len(text)
    
    if not font_char_counts:
        return None
    return max(font_char_counts, key=font_char_counts.get)


def extract_and_score_spans(pdf_path, bold_bonus=2.0):
    """Step 2: Extract text and calculate the Composite Score for every span."""
    doc = fitz.open(pdf_path)
    
    # Run Step 1 to get our anchor
    base_body_font = find_baseline_font_size(doc)
    print(f"[Step 1] Detected Base Body Font Size: {base_body_font}")
    
    scored_spans = []
    
    for page_num, page in enumerate(doc):
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        
                        font_size = round(span.get("size", 0.0), 1)
                        font_name = span.get("font", "").lower()
                        
                        # PyMuPDF uses flags to indicate styling. Bit 4 (value 16) is bold.
                        is_bold = bool(span.get("flags", 0) & 16) or "bold" in font_name
                        
                        # Calculate the Composite Score
                        composite_score = font_size + (bold_bonus if is_bold else 0.0)
                        
                        scored_spans.append({
                            "text": text,
                            "page": page_num + 1,
                            "font_size": font_size,
                            "is_bold": is_bold,
                            "composite_score": composite_score
                        })
                        
    doc.close()
    return base_body_font, scored_spans


def cluster_and_tag_hierarchy(scored_spans, base_body_font, tolerance=0.8):
    """Step 3: Cluster composite scores and assign H1, H2, Body, Footer tags."""
    
    # 1. Get all unique composite scores and sort them ascendingly
    unique_scores = sorted(list(set(span['composite_score'] for span in scored_spans)))
    
    if not unique_scores:
        return scored_spans, {}

    # 2. 1D Agglomerative Clustering
    clusters = []
    current_cluster = [unique_scores[0]]
    
    for score in unique_scores[1:]:
        # If the gap is within our tolerance, group them together
        if score - current_cluster[-1] <= tolerance:
            current_cluster.append(score)
        else:
            # Gap is too big -> save the cluster and start a new one
            clusters.append(current_cluster)
            current_cluster = [score]
    
    clusters.append(current_cluster)  # Append the final cluster

    # 3. Find which cluster contains the Base Body Font
    body_cluster_idx = 0
    for i, cluster in enumerate(clusters):
        # We check if the base_body_font falls into this cluster's range
        if any(abs(s - base_body_font) <= 0.1 for s in cluster):
            body_cluster_idx = i
            break

    # 4. Assign Tags (H1, H2, Body, Footer)
    score_to_tag = {}
    num_headers = len(clusters) - 1 - body_cluster_idx
    
    for i, cluster in enumerate(clusters):
        if i < body_cluster_idx:
            tag = "Footer"
        elif i == body_cluster_idx:
            tag = "Body"
        else:
            # Calculate header level (Highest score = H1)
            header_level = num_headers - (i - body_cluster_idx) + 1
            tag = f"H{header_level}"
            
        # Map every score in this cluster to the calculated tag
        for score in cluster:
            score_to_tag[score] = tag

    # 5. Apply the tags back to the original text spans
    for span in scored_spans:
        span['tag'] = score_to_tag[span['composite_score']]
        
    return scored_spans, score_to_tag


if __name__ == "__main__":
    pdf_file = "Shree.pdf"
    
    # Execute Step 1 & 2
    base_font, all_scored_text = extract_and_score_spans(pdf_file)
    
    # Execute Step 3
    final_spans, hierarchy_map = cluster_and_tag_hierarchy(all_scored_text, base_font)
    
    print(f"[Step 3] Hierarchy Map Generated: {hierarchy_map}")
    
    # Clean the data: Drop all footers and metadata before chunking
    clean_spans = [span for span in final_spans if span['tag'] != "Footer"]
    
    print(f"[Data Prep] Removed {len(final_spans) - len(clean_spans)} footer spans.")
    print(f"[Data Prep] Remaining structural spans for chunking: {len(clean_spans)}\n")
    
    # Print the first 10 valid structural spans to verify
    for span in clean_spans[:10]:
        print(f"[{span['tag']}] (Score: {span['composite_score']}) -> {span['text'][:50]}...")
    import json

    # At the end of your current extraction script:
    with open("structured_spans.json", "w", encoding="utf-8") as f:
        json.dump(clean_spans, f, indent=4)
    print("[Success] Saved structured spans to structured_spans.json")