# app/api.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from rag.rag_query import RAGEngine
import uvicorn

app = FastAPI(title="NW Realite RAG")
rag = RAGEngine(chroma_dir="data/chroma")

class QueryIn(BaseModel):
    q: str
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search")
def search(body: QueryIn):
    result = rag.answer(body.q, top_k=body.top_k)
    return result

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
