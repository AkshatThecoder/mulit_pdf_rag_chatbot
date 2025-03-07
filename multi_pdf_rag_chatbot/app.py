import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.llm_helper import get_llm
from dotenv import load_dotenv
import os

# Fix for Streamlit & Torch conflict
os.environ["STREAMLIT_WATCH_FILEWATCHER_TYPE"] = "none"

# Load env (for Groq key)
load_dotenv()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to process uploaded PDFs into FAISS
def process_uploaded_pdfs(files):
    all_docs = []
    for uploaded_file in files:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Attach file name as metadata
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_db")

    # Clean up temp files
    for uploaded_file in files:
        os.remove(f"temp_{uploaded_file.name}")

# Function to load FAISS retriever
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever()

# Function to get LLM answer with file-aware source info
def get_answer(query, retriever):
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    result = qa_chain.invoke({"query": query})

    # Collect source files
    sources = {doc.metadata.get("source", "Unknown file") for doc in result.get("source_documents", [])}
    sources_text = ", ".join(sources)

    full_response = f"{result['result']}\n\n(Sources: {sources_text})"
    return full_response

# Main Streamlit app
def main():
    st.set_page_config(page_title="Multi-PDF RAG Chatbot (Groq Powered)")
    st.title("Multi-PDF RAG Chatbot (Groq Powered)")

    # Sidebar for PDF Upload
    with st.sidebar:
        st.title("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True
        )
        if st.button("Process PDFs"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF!")
            else:
                process_uploaded_pdfs(uploaded_files)
                st.success("PDFs processed and stored in FAISS!")

    # Main chat interface
    user_question = st.text_input("Ask a question from the PDFs")

    if user_question:
        retriever = load_retriever()
        response = get_answer(user_question, retriever)

        # Append to chat history
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Bot", response))

    # Show chat history
    st.write("### Chat History")
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑‍💻 {role}:** {message}")
        else:
            st.markdown(f"**🤖 {role}:** {message}")

if __name__ == "__main__":
    main()
