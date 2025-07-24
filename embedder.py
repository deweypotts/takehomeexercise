"""
Document ingestion, splitting, and embedding logic for RAG pipeline.
"""
import os
import glob
import json
from typing import List, Tuple
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
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
    Splits documents into chunks that respect paragraph boundaries while maintaining character limits.
    Each paragraph becomes its own chunk unless it exceeds the character limit.
    Returns a list of dicts: {"source": filename, "content": chunk}
    """
    chunks = []
    
    for filename, text in docs:
        # Split text into paragraphs (double newlines)
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph is within character limit, keep it as one chunk
            if len(paragraph) <= config.CHUNK_SIZE:
                chunks.append({"source": filename, "content": paragraph})
            else:
                # If paragraph is too large, split it by sentences
                sentences = paragraph.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # If adding this sentence would exceed chunk size, save current chunk and start new one
                    if len(current_chunk) + len(sentence) > config.CHUNK_SIZE and current_chunk:
                        chunks.append({"source": filename, "content": current_chunk.strip() + "."})
                        current_chunk = sentence
                    else:
                        # Add sentence to current chunk
                        if current_chunk:
                            current_chunk += ". " + sentence
                        else:
                            current_chunk = sentence
                
                # Add the last chunk if it has content
                if current_chunk.strip():
                    chunks.append({"source": filename, "content": current_chunk.strip()})
    
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