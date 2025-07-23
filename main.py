"""
Main entry point for the modular RAG pipeline.
"""
import argparse
import config
from embedder import load_documents, split_documents, embed_chunks
from retriever import retrieve_context
from generator import construct_prompt, generate_response
from evaluator import batch_evaluate_csv

def cli_prompt_mode(vectorstore):
    """
    Allows user to enter a single prompt via CLI and prints the result and context.
    """
    user_prompt = input("Enter your prompt: ")
    context = retrieve_context(vectorstore, user_prompt)
    prompt = construct_prompt(config.SYSTEM_PROMPT, context, user_prompt)
    result = generate_response(prompt)
    print("\n--- Retrieved Context ---")
    for i, ctx in enumerate(context, 1):
        print(f"[{i}] {ctx}\n")
    print("--- AI Result ---")
    print(result)

def main():
    parser = argparse.ArgumentParser(description="Modular RAG System with OpenAI, FAISS, and LangChain")
    parser.add_argument("--docs_dir", type=str, default=config.DOCS_DIR, help=f"Directory containing documents (.txt, .md, .json) [default: {config.DOCS_DIR}]")
    parser.add_argument("--batch_csv", type=str, help="CSV file for batch evaluation (columns: user_prompt, result_truth)")
    parser.add_argument("--output_csv", type=str, default="output.csv", help="Output CSV for batch mode")
    parser.add_argument("--cli", action="store_true", help="Run in CLI single prompt mode")
    args = parser.parse_args()

    # Ingest and index documents
    print("Loading and indexing documents...")
    docs = load_documents(args.docs_dir)
    chunks = split_documents(docs)
    vectorstore, _ = embed_chunks(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")

    if args.batch_csv:
        batch_evaluate_csv(vectorstore, args.batch_csv, args.output_csv)
    elif args.cli:
        cli_prompt_mode(vectorstore)
    else:
        print("No mode selected. Use --cli for single prompt or --batch_csv for batch evaluation.")

if __name__ == "__main__":
    main()


"""
# CLI mode (will automatically use the 'docs' folder):
python main.py --cli

# Batch mode (will automatically use the 'docs' folder):
python main.py --batch_csv your_file.csv

# Or still specify a different folder if needed:
python main.py --docs_dir other_folder --cli


"""