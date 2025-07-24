"""
Test script to show the existing chunks from the embedder.
"""
import config
from embedder import load_documents, split_documents, embed_chunks

def inspect_text_structure():
    """
    Inspect the raw text structure to understand paragraph formatting.
    """
    print("Loading documents from:", config.DOCS_DIR)
    docs = load_documents(config.DOCS_DIR)
    
    for filename, text in docs:
        print(f"\n=== DOCUMENT: {filename} ===")
        print("First 1000 characters:")
        print("-" * 40)
        print(repr(text[:1000]))  # Show raw characters including newlines
        print("-" * 40)
        
        # Show paragraph splits
        paragraphs = text.split('\n\n')
        print(f"\nFound {len(paragraphs)} paragraphs using '\\n\\n' split")
        for i, para in enumerate(paragraphs[:3]):  # Show first 3 paragraphs
            print(f"\nParagraph {i+1}:")
            print(repr(para[:200]))  # Show first 200 chars with escape sequences

def test_existing_chunks():
    """
    Load documents, create chunks, and display the first 5 chunks.
    """
    print("Loading documents from:", config.DOCS_DIR)
    docs = load_documents(config.DOCS_DIR)
    print(f"Loaded {len(docs)} documents")
    
    print("\nSplitting documents into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    
    print("\n" + "="*80)
    print("FIRST 5 CHUNKS:")
    print("="*80)
    
    for i, chunk in enumerate(chunks[:5], 1):
        print(f"\n--- CHUNK {i} ---")
        print(f"Source: {chunk['source']}")
        print(f"Content length: {len(chunk['content'])} characters")
        print(f"Content preview:")
        print("-" * 40)
        print(chunk['content'])
        print("-" * 40)
        print()

if __name__ == "__main__":
    # First inspect the text structure
    inspect_text_structure()
    
    print("\n" + "="*80)
    print("NOW TESTING CHUNKS:")
    print("="*80)
    
    # Then test the chunks
    test_existing_chunks()