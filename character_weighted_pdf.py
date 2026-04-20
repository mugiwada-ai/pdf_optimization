import fitz  # PyMuPDF
from collections import defaultdict

def find_baseline_font_size(pdf_path):
    # Dictionary to hold {font_size: total_character_count}
    font_char_counts = defaultdict(int)
    
    # Open the PDF document
    doc = fitz.open(pdf_path)
    
    for page in doc:
        # Extract the page structure as a dictionary
        text_dict = page.get_text("dict")
        
        # Navigate the nested PDF structure: blocks -> lines -> spans
        for block in text_dict.get("blocks", []):
            # Process only text blocks (type 0), ignore images
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        # Clean the text and skip empty spans
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        
                        # Round font size to 1 decimal place
                        # (PDFs often render size 11 as 10.999 or 11.002)
                        font_size = round(span.get("size", 0.0), 1)
                        
                        # Add the character count to the corresponding font size
                        font_char_counts[font_size] += len(text)
    
    # If the PDF is completely empty or just images
    if not font_char_counts:
        return None, {}

    # Find the font size with the absolute highest character count
    base_body_font = max(font_char_counts, key=font_char_counts.get)
    
    return base_body_font, dict(font_char_counts)

base_font, distribution = find_baseline_font_size("temple_guide.pdf")
print(f"Detected Body Font Size: {base_font}")
print(f"Distribution: {distribution}")