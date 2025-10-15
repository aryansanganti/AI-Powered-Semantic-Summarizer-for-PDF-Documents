import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import os

_MODEL = None
_INDEX = None
_METADATA = None

def _ensure_resources(index_path="faiss_index.bin", metadata_path="metadata.json", model_name="all-MiniLM-L6-v2"):
    """Lazy-load model, index, and metadata. Auto-build if not found."""
    global _MODEL, _INDEX, _METADATA
    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        from store_embeddings import build_index
        print("No FAISS index found. Building from PDFs in current directory...")
        build_index(index_path=index_path, metadata_path=metadata_path)
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    if _INDEX is None:
        _INDEX = faiss.read_index(index_path)
    if _METADATA is None:
        with open(metadata_path, "r", encoding="utf-8") as f:
            _METADATA = json.load(f)

def search_faiss(query, k=3):
    """Search for the most relevant text snippets from the FAISS index.
    
    Args:
        query (str): The search query.
        k (int): The number of top matches to retrieve.
    
    Returns:
        str: A single string combining the top `k` matching text snippets.
    """
    if not query or query.strip().lower() == "exit":
        return ""
    _ensure_resources()
    # Convert query to embedding
    query_embedding = _MODEL.encode([query])
    # Bound k
    ntotal = _INDEX.ntotal
    k = max(1, min(int(k), ntotal))
    # Search in FAISS index
    _, indices = _INDEX.search(np.array(query_embedding, dtype=np.float32), k)
    # Retrieve best matches and combine into a single string
    best_matches = "\n".join([_METADATA[idx]["text"] for idx in indices[0]])
    return best_matches
