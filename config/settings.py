from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Models
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# LLM Parameters
TEMPERATURE = 0
MAX_TOKENS = 100
