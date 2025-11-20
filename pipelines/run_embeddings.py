# pipelines/run_embeddings.py
import os
import uuid
from rag.embed import load_embed_model, embed_texts
from rag.vector_store import get_client, create_or_get_collection, upsert_documents
from utils.chunker import chunk_text_from_file

def build_embeddings_for_textfile(txt_path: str, chroma_dir="data/chroma"):
    # read text, chunk it
    chunks = chunk_text_from_file(txt_path)
    ids = [str(uuid.uuid4()) for _ in chunks]
    model = load_embed_model()
    embs = embed_texts(model, chunks)
    client = get_client(chroma_dir)
    collection = create_or_get_collection(client, name="valuation_reports")
    upsert_documents(collection, ids=ids, documents=chunks, embeddings=embs)
    print(f"Indexed {len(chunks)} chunks to ChromaDB at {chroma_dir}")
    return len(chunks)

if __name__ == "__main__":
    txt_path = "data/ocr/sample_output.txt"
    build_embeddings_for_textfile(txt_path)
