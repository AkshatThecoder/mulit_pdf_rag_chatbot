from langchain_groq import ChatGroq
import os

def get_llm():
    return ChatGroq(
        api_key=os.getenv("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"),
        model_name="llama3-70b-8192"

    )
