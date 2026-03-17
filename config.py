import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
FAISS_INDEX_PATH = "vectorstore/faiss_index"
UPLOAD_PATH = "data/upload_docs"
TOP_K_RESULTS = 3

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ✅ reads from .env