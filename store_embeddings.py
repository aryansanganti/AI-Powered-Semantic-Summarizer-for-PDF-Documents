import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from break_text import chunk_text
from get_text import extract_text_from_pdf
import os
from glob import glob

def build_index(pdf_files=None, index_path="faiss_index.bin", metadata_path="metadata.json", model_name="all-MiniLM-L6-v2"):
    """Build and save FAISS index and metadata from given PDFs (or auto-detected PDFs)."""
    # Auto-discover PDFs if none provided
    if not pdf_files:
        pdf_files = sorted(glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the current directory.")

    # Initialize model locally to avoid import-time cost
    model = SentenceTransformer(model_name)

    all_texts = []
    metadata = []

    for pdf in pdf_files:
        raw_text = extract_text_from_pdf(pdf)
        if not raw_text:
            continue
        texts = chunk_text(raw_text)
        all_texts.extend(texts)
        # Store metadata with text index
        for text in texts:
            metadata.append({"pdf": pdf, "text": text})

    if not all_texts:
        raise ValueError("No text extracted from the provided PDFs.")

    # Encode in batches with progress bar
    embeddings = model.encode(
        all_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.asarray(embeddings, dtype=np.float32))

    # Save index and metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"Indexed {len(all_texts)} chunks from {len(pdf_files)} PDFs.")
    print(f"Saved index -> {index_path}")
    print(f"Saved metadata -> {metadata_path}")

if __name__ == "__main__":
    # Build index from all PDFs in the current directory
    build_index()
