import sys
sys.path.append('src')
sys.path.append('.')

from src.ingestion import load_and_split
from src.vectorstore import build_vectorstore, load_vectorstore, vectorstore_exists

# Load test document
chunks = load_and_split("data/upload_docs/test.txt")

# Build and save index
vectorstore = build_vectorstore(chunks)

# Check it exists
print(f"✅ Index exists: {vectorstore_exists()}")

# Reload from disk
vectorstore2 = load_vectorstore()

# Test a search
results = vectorstore2.similarity_search("test document", k=1)
print(f"✅ Search result: {results[0].page_content}")
