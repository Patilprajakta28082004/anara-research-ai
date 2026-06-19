"""
Anara ResearchAI — Corrective RAG (CRAG) API.

Endpoints:
  GET  /                      health check
  POST /api/rag/ingest        upload a PDF -> chunk -> embed -> store in ChromaDB
  POST /api/rag/query         ask a question, answered via the CRAG graph
  GET  /api/research/search   search arXiv + Semantic Scholar
"""
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.services.finder import search_literature
from app.services.graph import run_crag_pipeline
from app.services.ingest import ingest_pdf

app = FastAPI(
    title="Anara ResearchAI — Corrective RAG API",
    description="Upload research PDFs, ask grounded questions, and search arXiv / Semantic Scholar.",
    version="1.0.0",
)

# CORS — wide open for local dev; restrict ALLOWED_ORIGINS in production via .env
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to ask over ingested PDFs")


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "message": "CRAG API is running"}


# --------------------------------------------------------------------------
# RAG: ingest
# --------------------------------------------------------------------------
@app.post("/api/rag/ingest")
async def ingest_document(file: UploadFile = File(...)):
    return await ingest_pdf(file)


# --------------------------------------------------------------------------
# RAG: query
# --------------------------------------------------------------------------
@app.post("/api/rag/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    try:
        return run_crag_pipeline(payload.query)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer query: {exc}")


# --------------------------------------------------------------------------
# Research: literature search
# --------------------------------------------------------------------------
@app.get("/api/research/search")
async def research_endpoint(query: str, limit: int = 10):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    results = await search_literature(query, limit=limit)
    return {"query": query, "count": len(results), "results": results}
