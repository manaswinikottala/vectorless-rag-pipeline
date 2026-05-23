# 📄 RAG Pipeline for PDF Memos

A local Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions
about your PDF memos using **Claude** as the LLM and **ChromaDB** as the vector store.

---

## Stack

| Layer       | Tool                        |
|-------------|-----------------------------|
| PDF parsing | PyMuPDF (`fitz`)            |
| Embeddings  | `sentence-transformers` (local, no API cost) |
| Vector DB   | ChromaDB (persistent, local) |
| LLM         | Anthropic Claude (Sonnet)   |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

### Step 1 — Ingest your PDF memos

Point the script at a folder of PDFs. It will parse, chunk, embed, and store them.

```bash
python rag.py ingest /path/to/your/memos/
```

You only need to do this once (or again when you add new PDFs).
The vector DB is saved to `./chroma_db/` by default.

---

### Step 2 — Ask a single question

```bash
python rag.py ask "What did we decide about the Q3 budget?"
```

---

### Step 3 — Interactive chat loop

```bash
python rag.py chat
```

Type your questions one by one. Type `quit` to exit.

---

## Options

| Flag       | Default        | Description                   |
|------------|----------------|-------------------------------|
| `--db`     | `./chroma_db`  | Path to ChromaDB storage      |

Example with custom DB path:

```bash
python rag.py ingest ./memos --db ./my_vector_db
python rag.py chat   --db ./my_vector_db
```

---

## How it works

```
PDF files
   ↓  PyMuPDF
Plain text
   ↓  Chunking (500 chars, 100 overlap)
Text chunks
   ↓  sentence-transformers (local embedding)
ChromaDB (vector store)
   ↑  Cosine similarity search (top 5 chunks)
Query → Retrieved context
   ↓  Anthropic Claude
Answer + Sources
```

---

## Tips

- **Re-ingest anytime** — `upsert` is used so duplicate chunks are safely overwritten.
- **Tune chunk size** — Edit `CHUNK_SIZE` and `CHUNK_OVERLAP` in `rag.py` for your memo length.
- **More context** — Increase `TOP_K` if answers feel incomplete.