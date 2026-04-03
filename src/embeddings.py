# src/embeddings.py

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL

def get_embeddings():
    """Load and return the HuggingFace embedding model."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"✅ Embedding model loaded: {EMBEDDING_MODEL}")
    return embeddings