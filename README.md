

### How this pipeline works
```
PDF → Images → OCR → text → chunk → embed → Chroma → retrieve → Phi-3 → answer
```

### Project Structure
```
nw-rag-mvp/
│
├── data/
│   ├── pdfs/                 # Raw scanned valuation PDFs
│   ├── images/               # PDF pages as images for OCR
│   ├── ocr/                  # OCR text files
│   ├── chunks/               # Chunked text from each report
│   └── chroma/               # Persistent ChromaDB vector store
│
├── ocr/
│   ├── pdf_to_images.py      # Convert PDF -> images
│   └── extract_text.py       # PaddleOCR wrapper to extract text
│
├── rag/
│   ├── embed.py              # Embedding generation (BGE-small)
│   ├── vector_store.py       # ChromaDB interface
│   ├── phi3_loader.py        # Load Phi-3 Mini model (CPU)
│   └── rag_query.py          # Retrieval + LLM answer generation
│
├── app/
│   └── api.py                # FastAPI ( /search endpoint )
│
├── pipelines/
│   ├── run_ocr.py            # PDF → images → text
│   ├── run_embeddings.py     # Text → chunks → embeddings → ChromaDB
│   └── run_all.py            # Complete pipeline (OCR + Embed)
│
├── utils/
│   ├── chunker.py            # Text chunking logic
│   ├── file_utils.py         # File helpers
│   └── logger.py             # Simple logging utility
│
├── models/                   # (Optional) future LLMs
│   └── README.md
│
├── config/
│   ├── model_config.yaml     # Which embedding & LLM to use
│   ├── paths.yaml            # Folder paths (data/, ocr/, chunks/)
│   └── chroma.yaml           # Vector DB configs
│
├── requirements.txt
└── README.md

```