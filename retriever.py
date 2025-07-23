"""
Retriever logic for RAG pipeline using FAISS.
"""
from typing import List
import config

def retrieve_context(vectorstore, query: str, k: int = None) -> List[str]:
    """
    Retrieves the top-k most relevant chunks from the FAISS index.
    """
    if k is None:
        k = config.TOP_K
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs] 