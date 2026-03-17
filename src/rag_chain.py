# src/rag_chain.py

from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from config import LLM_MODEL, TOP_K_RESULTS

def load_llm():
    """Load the local LLM (FLAN-T5)."""
    pipe = pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_new_tokens=256,
        temperature=0.3    # lower = more focused answers
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"✅ LLM loaded: {LLM_MODEL}")
    return llm

def build_rag_chain(vectorstore):
    """Build the RAG chain using vectorstore + LLM."""
    llm = load_llm()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K_RESULTS}
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # stuffs all chunks into one prompt
        retriever=retriever,
        return_source_documents=True # enables citations
    )
    return chain

def get_answer(chain, question: str):
    """Ask a question and return answer + sources."""
    result = chain({"query": question})
    answer = result["result"]
    sources = [
        doc.metadata.get("source", "Unknown")
        for doc in result["source_documents"]
    ]
    return answer, list(set(sources))