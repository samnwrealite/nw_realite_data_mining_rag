# rag/vector_store.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import os

CHROMA_PERSIST_DIR = "data/chroma"

def get_client(persist_directory: str = CHROMA_PERSIST_DIR):
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)
    return client

def create_or_get_collection(client, name="valuation_reports"):
    return client.get_or_create_collection(name)

def upsert_documents(collection, ids: List[str], documents: List[str], embeddings):
    """
    ids: list of string ids
    documents: list of texts
    embeddings: numpy array or list of lists
    """
    collection.add(ids=ids, documents=documents, embeddings=embeddings)
    return True

def query_collection(collection, query_embeddings, n_results=5):
    res = collection.query(query_embeddings=query_embeddings, n_results=n_results, include=["documents", "scores", "metadatas"])
    return res
