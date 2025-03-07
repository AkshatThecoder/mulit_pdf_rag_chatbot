# Multi-PDF RAG Chatbot (Groq Powered)

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that lets you ask questions across **multiple PDFs**. It combines **Langchain**, **FAISS**, and **Groq's Llama 3** to give you accurate answers directly from your uploaded documents.

## 📂 Folder Structure

## ⚡️ What’s New in app.py (Upgraded Version)

| Feature | Description |
|---|---|
| 📤 Direct PDF Upload | Upload PDFs directly via Streamlit sidebar, no need to run ingest.py separately |
| 📥 Automatic Processing | PDF is chunked, embedded & stored into FAISS automatically |
| 💬 Chat History | Full conversation history shown in UI |
| 📄 Source-aware Answers | Every answer tells you **which PDF(s)** the information came from |
| 🚀 Simplified Flow | No manual preprocessing needed — fully handled inside the UI |

## 💻 How to Run (New Flow)

1. Install requirements:
```pip install -r `requirements.txt```
2. Create a .env file in the root with:
```GROQ_API_KEY=your-actual-groq-key```
3. Start the chatbot:
```streamlit run app.py```
4. Upload PDFs directly in the sidebar.
5. Ask questions in the chat box.


##  🗃️ What About old_app.py and ingest.py? 

| File | Purpose |
|---|---|
| old_app.py | The original version, which required you to manually run ingest.py first |
| ingest.py	| Standalone PDF processor that creates faiss_db — still works if you want to process PDFs outside Streamlit |
