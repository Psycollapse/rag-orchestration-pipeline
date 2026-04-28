import os
from dotenv import load_dotenv

load_dotenv()

# --- API KEYS ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- PINECONE ---
INDEX_NAME = "rag-pipeline"
DIMENSION = 1536
METRIC = "cosine"

# --- MODEL ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"