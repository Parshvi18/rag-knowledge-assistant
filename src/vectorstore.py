# src/vectorstore.py

import os
from langchain_community.vectorstores import FAISS
from embeddings import get_embeddings 
from config import FAISS_INDEX_PATH

def build_vectorstore(chunks):
    """Build FAISS index from document chunks and save it."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save to disk so you don't rebuild every time
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"✅ Vector store saved to {FAISS_INDEX_PATH}")
    return vectorstore

def load_vectorstore():
    """Load existing FAISS index from disk."""
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded from disk")
    return vectorstore

def vectorstore_exists():
    """Check if a saved index already exists."""
    return os.path.exists(FAISS_INDEX_PATH)