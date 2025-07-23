"""
Document ingestion, splitting, and embedding logic for RAG pipeline.
"""
import os
import glob
import json
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import config

def load_documents(directory: str) -> List[Tuple[str, str]]: ## loads the docs
    """
    Loads all .txt, .md, and .json files from a directory.
    Returns a list of (filename, text) tuples.
    """
    docs = []
    for ext in ("*.txt", "*.md", "*.json"):
        for filepath in glob.glob(os.path.join(directory, ext)):
            with open(filepath, "r", encoding="utf-8") as f:
                if filepath.endswith(".json"):
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            docs.append((filepath, item.get("text", "")))
                    elif isinstance(data, dict):
                        docs.append((filepath, data.get("text", "")))
                else:
                    docs.append((filepath, f.read()))
    return docs

def split_documents(docs: List[Tuple[str, str]]) -> List[dict]: ## puts the loaded docs into chunks (chunk size configured in the config.py file)
    """
    Splits documents into chunks using LangChain's RecursiveCharacterTextSplitter.
    Returns a list of dicts: {"source": filename, "content": chunk}
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = []
    for filename, text in docs:
        for chunk in splitter.split_text(text):
            chunks.append({"source": filename, "content": chunk})
    return chunks

def embed_chunks(chunks: List[dict]): ## turns the chunks into vectors and adds chunk metadata
    """
    Embeds chunks using OpenAI and stores them in a FAISS vector store.
    Returns the FAISS index and the list of chunks (for metadata).
    """
    embeddings = OpenAIEmbeddings(model=config.EMBED_MODEL, openai_api_key=config.OPENAI_API_KEY)
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore, chunks 