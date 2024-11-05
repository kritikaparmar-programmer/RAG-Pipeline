import streamlit as st
import asyncio
import nest_asyncio
from rag import RAG  # Import your updated RAG module

# Apply nest_asyncio to support nested event loops
nest_asyncio.apply()

# Initialize Streamlit session state
st.session_state.clicked = False
vectorstore_created = False

@st.cache_resource(show_spinner=True)
def load_rag_pipeline(web_path):
    """Load the RAG pipeline with the specified web path."""
    return RAG(web_path)

# Streamlit App UI
st.title("RAG")
st.subheader("RAG pipeline for website questioning")

# Input for website URL
web_path = st.sidebar.text_input("Enter website URL")
if web_path:
    rag_pipe = load_rag_pipeline(web_path)
    st.session_state.clicked = True

# Question input and response retrieval
if st.session_state.clicked:
    question = st.text_input("Enter your question")
    if question:
        out, vs = rag_pipe.qa(question, vectorstore_created)
        vectorstore_created = vs
        st.write(out)
