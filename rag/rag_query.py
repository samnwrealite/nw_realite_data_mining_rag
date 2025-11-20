# rag/rag_query.py
from typing import List, Dict, Any
import numpy as np
from .vector_store import get_client, create_or_get_collection, query_collection
from .embed import load_embed_model, embed_texts
from .phi3_loader import load_phi3_model, generate_answer

# load models lazily to keep import light
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PHI3_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

class RAGEngine:
    def __init__(self, chroma_dir="data/chroma"):
        self.chroma_dir = chroma_dir
        self.client = get_client(chroma_dir)
        self.collection = create_or_get_collection(self.client, name="valuation_reports")
        self.embed_model = load_embed_model(EMBED_MODEL_NAME)
        # Phi3 loading can be slow; keep it optional
        try:
            self.tokenizer, self.phi3 = load_phi3_model(PHI3_MODEL_NAME, device="cpu")
            self.has_llm = True
        except Exception as e:
            print("Warning: Phi-3 model failed to load (will fallback to returning context).", e)
            self.tokenizer, self.phi3 = None, None
            self.has_llm = False

    def retrieve(self, query: str, top_k: int = 5):
        q_emb = embed_texts(self.embed_model, [query])
        res = query_collection(self.collection, query_embeddings=q_emb.tolist(), n_results=top_k)
        # res structure: dict with 'ids','documents','distances' or similar depending on chroma version
        return res

    def answer(self, query: str, top_k: int = 5):
        retrieval = self.retrieve(query, top_k=top_k)
        docs = retrieval.get("documents", [[]])[0]
        scores = retrieval.get("scores", [[]])[0] if "scores" in retrieval else []
        metadatas = retrieval.get("metadatas", [[]])[0] if "metadatas" in retrieval else []

        # Build context from top documents
        context_parts = []
        for i, doc in enumerate(docs):
            meta = metadatas[i] if i < len(metadatas) else {}
            header = f"[source:{meta.get('source','unknown')} score:{scores[i] if i < len(scores) else 'N/A'}]"
            context_parts.append(header + "\n" + doc)
        context = "\n\n".join(context_parts)

        if self.has_llm:
            prompt = f"""You are a concise assistant. Use ONLY the context supplied to answer the question.
CONTEXT:
{context}

QUESTION:
{query}

Answer briefly and include citations in the form (source).
"""
            try:
                out = generate_answer(self.tokenizer, self.phi3, prompt, max_new_tokens=200)
                return {"answer": out, "context": context, "retrieval": {"docs": docs, "scores": scores, "metadatas": metadatas}}
            except Exception as e:
                # fallback
                return {"answer": context[:1000], "context": context, "retrieval": {"docs": docs, "scores": scores, "metadatas": metadatas}, "error": str(e)}
        else:
            # no LLM available -> return top-k context to the caller
            return {"answer": context, "context": context, "retrieval": {"docs": docs, "scores": scores, "metadatas": metadatas}, "note": "LLM not loaded; returning context only."}
