import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from config import LLM_MODEL, TOP_K_RESULTS

def load_llm():
    """Load the local LLM (FLAN-T5)."""
    pipe = pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_new_tokens=256,
        do_sample=False,
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

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
Context: {context}
Question: {question}
Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def get_answer(chain, question: str):
    """Ask a question and return answer + sources."""
    answer = chain.invoke(question)
    return answer, []

if __name__ == "__main__":
    print("✅ RAG chain module loaded OK")