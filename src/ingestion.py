# src/ingestion.py

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_document(file_path: str):
    """Load a PDF or text file and return raw documents."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Only PDF and TXT files are supported.")
    
    return loader.load()

def split_documents(documents):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    return chunks

def load_and_split(file_path: str):
    """Main function: load + split in one step."""
    docs = load_document(file_path)
    chunks = split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks from {file_path}")
    return chunks