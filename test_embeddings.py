import sys
sys.path.append('..')

from src.embeddings import get_embeddings

# Load the model
embeddings = get_embeddings()

# Test with sample sentences
texts = [
    "Python is a programming language",
    "Python is used for data science",
    "I love eating pizza"
]

vectors = embeddings.embed_documents(texts)

print(f"\n✅ Number of texts embedded: {len(vectors)}")
print(f"✅ Vector size: {len(vectors[0])} dimensions")
print(f"✅ First vector preview: {vectors[0][:5]}...")

# Test similarity (similar texts should have closer scores)
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

sim1 = cosine_similarity(vectors[0], vectors[1])  # both about Python
sim2 = cosine_similarity(vectors[0], vectors[2])  # Python vs pizza

print(f"\n📊 Similarity (Python vs Python): {sim1:.4f}")  # should be HIGH
print(f"📊 Similarity (Python vs Pizza):  {sim2:.4f}")  # should be LOW
