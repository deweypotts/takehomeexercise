"""
Configuration constants for the RAG pipeline.
"""
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key as an environment variable
EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"  # or "gpt-4"
SYSTEM_PROMPT = "You are a helpful assistant. Use the provided context to answer the user's question as accurately as possible."
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
DOCS_DIR = "docs"  # Default documents directory

#note: temp is in teh generator file