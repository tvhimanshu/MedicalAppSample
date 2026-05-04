import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    VECTOR_SEARCH_URL      = os.getenv("VECTOR_SEARCH_URL",      "http://localhost:4000")
    VECTOR_SEARCH_EMAIL    = os.getenv("VECTOR_SEARCH_EMAIL",    "medrag@demo.com")
    VECTOR_SEARCH_PASSWORD = os.getenv("VECTOR_SEARCH_PASSWORD", "MedRag@2024!")

    OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY",         "")
    OPENAI_MODEL           = os.getenv("OPENAI_MODEL",           "gpt-4o-mini")
    OPENAI_EMBED_MODEL     = os.getenv("OPENAI_EMBED_MODEL",     "text-embedding-3-small")

    RAG_TOP_K              = int(os.getenv("RAG_TOP_K",          "6"))
    RAG_MIN_SCORE          = float(os.getenv("RAG_MIN_SCORE",    "0.30"))
