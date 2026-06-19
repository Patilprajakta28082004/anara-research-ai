"""
ChromaDB accessor.

A single, cached Chroma client + embedding function so we don't pay the
cost of re-loading the HuggingFace embedding model on every request.
"""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """Lazily build the embedding model once and reuse it (it's not free to load)."""
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_chroma_db() -> Chroma:
    """Returns a (cached) client for the ChromaDB collection."""
    return Chroma(
        collection_name="research_chunks",
        embedding_function=get_embedding_function(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
