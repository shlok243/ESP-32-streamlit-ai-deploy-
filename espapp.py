import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os

st.set_page_config(page_title="ESP32 Datasheet RAG", page_icon="📘")

st.title("📘 ESP32 Datasheet Question Answering")
st.write("Upload the ESP32 Datasheet PDF and ask questions.")

uploaded_file = st.file_uploader(
    "Upload ESP32 Datasheet",
    type=["pdf"]
)

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("Loading PDF..."):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    st.success(f"Loaded {len(documents)} pages.")

    with st.spinner("Splitting document..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )
        docs = splitter.split_documents(documents)

    st.success(f"Created {len(docs)} chunks.")

    with st.spinner("Loading Embedding Model..."):
        embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    with st.spinner("Creating FAISS Vector Store..."):
        db = FAISS.from_documents(docs, embedding)

    st.success("Vector Database Ready!")

    query = st.text_input("Ask a question about the ESP32 Datasheet")

    if st.button("Search") and query:

        results = db.similarity_search(query, k=3)

        st.subheader("Top Relevant Chunks")

        for i, result in enumerate(results, start=1):
            st.markdown(f"### Result {i}")
            st.write(result.page_content)
            st.divider()

    os.remove(pdf_path)