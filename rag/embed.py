# rag/embed.py
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# configurable model: change if you prefer another CPU-friendly model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fallback small model
# or "BAAI/bge-small-en-v1.5" if available locally

def load_embed_model(model_name: str = EMBED_MODEL):
    model = SentenceTransformer(model_name)
    return model

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Returns a 2D numpy array of embeddings (float32)
    """
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")
