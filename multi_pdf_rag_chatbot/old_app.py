import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from utils.llm_helper import get_llm
import os

# Fix for Streamlit + PyTorch conflict
os.environ["STREAMLIT_WATCH_FILEWATCHER_TYPE"] = "none"

load_dotenv()
import os
print(f"GROQ_API_KEY loaded: {os.getenv('GROQ_API_KEY')}")


def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever()

def get_answer(query, retriever):
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    result = qa_chain.invoke({"query": query})
    return result["result"]

def main():
    st.set_page_config(page_title="Multi-PDF RAG Chatbot (Groq Powered)")
    st.title("Multi-PDF RAG Chatbot (Groq Powered)")

    query = st.text_input("Ask your question")

    if query:
        retriever = load_retriever()
        answer = get_answer(query, retriever)
        st.write("Answer:", answer)

    with st.sidebar:
        st.write("First, run `ingest.py` to process PDFs placed in `pdfs/` folder.")

if __name__ == "__main__":
    main()
