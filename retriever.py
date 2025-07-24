"""
Retriever logic for RAG pipeline using FAISS.
"""
from typing import List, Tuple, Optional
import config

def retrieve_context(vectorstore, query: str, k: int = None, return_metadata: bool = False) -> List[str] | Tuple[List[str], List[dict]]:
    """
    Retrieves the top-k most relevant chunks from the FAISS index.
    
    Args:
        vectorstore: FAISS vector store
        query: Search query
        k: Number of chunks to retrieve (default from config)
        return_metadata: Whether to return metadata along with content
    
    Returns:
        If return_metadata=False: List of chunk contents
        If return_metadata=True: Tuple of (chunk_contents, metadata_list)
    """
    if k is None:
        k = config.TOP_K
    
    if return_metadata:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
        contents = [doc.page_content for doc, _ in docs_with_scores]
        metadata = [doc.metadata for doc, _ in docs_with_scores]
        return contents, metadata
    else:
        docs = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs] 