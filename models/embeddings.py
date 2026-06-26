from langchain_huggingface import HuggingFaceEndpointEmbeddings
from config.settings import EMBEDDING_MODEL

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-large-en-v1.5"
)