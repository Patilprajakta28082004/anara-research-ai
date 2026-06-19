# Anara ResearchAI — Corrective RAG (CRAG) API

A small research assistant: upload PDFs, ask grounded questions over them
via a **Corrective RAG** pipeline (LangGraph), and search live literature
on arXiv + Semantic Scholar.

## Architecture

```
frontend/index.html  ──HTTP──>  FastAPI app (app/main.py)
                                   ├── app/services/ingest.py   PDF -> chunks -> ChromaDB
                                   ├── app/services/graph.py    CRAG graph (LangGraph)
                                   ├── app/services/finder.py   arXiv + Semantic Scholar
                                   └── app/db/chroma.py         vector store accessor
```

**CRAG flow** (`graph.py`): `retrieve → grade_documents → (web_search if irrelevant) → generate`.
Retrieved chunks are graded for relevance by an LLM; if nothing relevant
comes back, a fallback node stands in for a live web search before
generation, instead of confidently answering from bad context.

## What was broken in the original drop, and the fix

| Problem | Fix |
|---|---|
| `main.py` imported `query_rag` from `finder.py` and `research_search` from `graph.py` — neither function exists in those files | Wired the real functions: `run_crag_pipeline` (graph) and `search_literature` (finder) |
| `/api/rag/ingest` called `ingest_pdf(file)` synchronously on a `UploadFile`, but `ingest_pdf` expected raw `bytes` | `ingest_pdf` is now `async`, reads the upload itself, validates type/size, and returns a useful JSON summary instead of a bare int |
| Inconsistent imports (`from chroma import ...` vs `from app.db.chroma import ...`) — the project wasn't actually one Python package | Restructured into a proper `app/` package with `db/` and `services/` modules so every import path is consistent |
| `graph.py` and `chroma.py` instantiated 3 LLMs and the embedding model on every single request | Cached with `functools.lru_cache`; LLM fallback chain is now only built from providers that actually have an API key set, so the demo runs with just one free key (Groq recommended) |
| arXiv XML parsing assumed every field always exists (`.find(...).text` with no `None` check) → would crash on malformed entries | Added a safe `_text_or_default` helper |
| No input validation anywhere (empty query, non-PDF upload, oversized file) | Added proper `HTTPException`s with real status codes (400/413/422) |
| Frontend `frontend-test.html` was functionally fine but visually bare and gave no loading/empty/error states | Rebuilt as `frontend/index.html`: tabbed "research desk" UI, drag-and-drop PDF upload, live API health indicator, loading/error/empty states, and literature results rendered as citation cards |

## Running it locally

```bash
cd anara-research-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add at least ONE key (GROQ_API_KEY is the easiest free tier)

uvicorn app.main:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (it talks to
`http://localhost:8000`).

## Things worth knowing 

- **Why CRAG instead of plain RAG**: plain RAG generates from whatever it
  retrieves, even if it's irrelevant. CRAG adds a grading step so the
  pipeline self-corrects (falls back to a web search) when local context
  doesn't actually answer the question — it's a guard against
  hallucinating off bad retrieval.
- **Why LangGraph**: the pipeline is a state machine, not a linear
  chain — `grade_documents` branches conditionally to either
  `web_search` or `generate`. LangGraph models that explicitly as a graph
  with typed state (`GraphState`), which is easier to reason about and
  extend than nested if/else chains.
- **Why an LLM fallback chain**: `with_fallbacks([...])` means if the
  primary provider errors or rate-limits, the call transparently retries
  on the next provider — useful resilience for a demo running on free-tier
  keys.
- **Why `lru_cache` on `get_chroma_db()` / the embedding model**: loading
  a sentence-transformers model is expensive; without caching, every
  request would reload it from disk.
- **Where this is intentionally simplified**: `web_search_fallback` is a
  mocked node (clearly labelled in the code) — swapping it for a real
  search API (Tavily/Serper/Bing) is a one-function change and a good
  "what would you do next" answer.
