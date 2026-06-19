"""
PDF ingestion pipeline: PDF bytes -> extracted text -> chunks -> ChromaDB.
"""
import uuid
from typing import Optional

import fitz  # PyMuPDF
from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.db.chroma import get_chroma_db


async def ingest_pdf(file: UploadFile, paper_id: Optional[str] = None) -> dict:
    """Read an uploaded PDF, extract text, chunk it, and store it in ChromaDB.

    Returns a small summary dict instead of a bare int so the frontend has
    something useful to render (filename, chunk count, doc id).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    file_content = await file.read()

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(file_content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.MAX_UPLOAD_MB}MB upload limit.",
        )

    if not file_content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Extract text
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
    except Exception as exc:  # malformed / encrypted PDF, etc.
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}")

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text found (the PDF may be a scanned image).",
        )

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_text(text)

    # 3. Build Document objects
    doc_id = paper_id or str(uuid.uuid4())
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": file.filename,
                "chunk_index": i,
                "paper_id": doc_id,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    # 4. Ingest into ChromaDB
    db = get_chroma_db()
    db.add_documents(documents)

    return {
        "filename": file.filename,
        "paper_id": doc_id,
        "chunks_indexed": len(documents),
    }
