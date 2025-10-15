from get_text import extract_text_from_pdf
import re

def chunk_text(text, chunk_size=1000):
    """Splits text into smaller chunks of approximately `chunk_size` characters, respecting sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split by sentence boundaries
    chunks, current_chunk = [], []

    for sentence in sentences:
        if sum(len(s) for s in current_chunk) + len(sentence) < chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Add the last chunk
    
    return chunks

