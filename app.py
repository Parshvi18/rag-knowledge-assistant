# app.py

import streamlit as st
from src.ingestion import load_and_split
from src.vectorstore import build_vectorstore, load_vectorstore, vectorstore_exists
from src.rag_chain import build_rag_chain, get_answer
from src.utils import save_uploaded_file, format_sources

# --- Page config ---
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG Knowledge Assistant")
st.caption("Upload a document and ask questions about it.")

# --- Upload section ---
uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt"],
    help="Supported formats: PDF, TXT"
)

if uploaded_file:
    with st.spinner("Processing your document..."):
        # Step 1: Save file
        file_path = save_uploaded_file(uploaded_file)
        
        # Step 2: Load and chunk
        chunks = load_and_split(file_path)
        
        # Step 3: Build vector store
        vectorstore = build_vectorstore(chunks)
        
        # Step 4: Build RAG chain and store in session
        st.session_state.chain = build_rag_chain(vectorstore)
        st.session_state.doc_name = uploaded_file.name

    st.success(f"✅ Ready! Indexed {len(chunks)} chunks from **{uploaded_file.name}**")

# --- Q&A section ---
if "chain" in st.session_state:
    st.divider()
    st.subheader(f"Ask about: {st.session_state.doc_name}")
    
    question = st.text_input(
        "Your question",
        placeholder="e.g. What is this document about?"
    )
    
    if question:
        with st.spinner("Finding answer..."):
            answer, sources = get_answer(st.session_state.chain, question)
        
        st.markdown("### Answer")
        st.write(answer)
        
        st.markdown("### Sources")
        st.text(format_sources(sources))
else:
    st.info("👆 Upload a document to get started.")