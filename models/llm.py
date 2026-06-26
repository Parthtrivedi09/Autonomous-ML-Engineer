from langchain_groq import ChatGroq
from  config.settings import LLM_MODEL
llm=ChatGroq(
    model_name="llama-3.3-70b-versatile",
    max_tokens=100
)