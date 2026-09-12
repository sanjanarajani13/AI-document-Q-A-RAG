# app.py

import streamlit as st
import os
from app.pipeline import process_pdf, answer_question

st.set_page_config(page_title="AI Document Q&A", page_icon="📄")

st.title("📄 AI-Powered Document Q&A")
st.write("Upload a PDF and ask questions about its contents.")

# Store the vector store in session state so it persists across interactions
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.filename = None

# --- Upload section ---
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
    save_path = os.path.join("data", "uploads", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document..."):
        st.session_state.vector_store = process_pdf(save_path)
        st.session_state.filename = uploaded_file.name

    st.success(f"'{uploaded_file.name}' processed and ready!")

# --- Question section ---
if st.session_state.vector_store is not None:
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(st.session_state.vector_store, question)
        st.markdown("### Answer")
        st.write(answer)