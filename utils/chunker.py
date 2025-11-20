# utils/chunker.py
from typing import List
import os

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end].strip())
        i += chunk_size - overlap
    return chunks

def chunk_text_from_file(path: str, chunk_size: int = 800, overlap: int = 100):
    with open(path, "r", encoding="utf8") as f:
        text = f.read()
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)
