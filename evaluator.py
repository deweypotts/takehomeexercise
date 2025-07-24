"""
Batch evaluation logic for RAG pipeline using CSV files.
"""
import pandas as pd
from tqdm import tqdm
import os
import config
from retriever import retrieve_context
from generator import construct_prompt, generate_response

def batch_evaluate_csv(vectorstore, csv_path: str, output_csv: str = None):
    """
    Loads a CSV with columns: user_prompt, result_truth.
    For each row, generates a result and prints it.
    Saves a new CSV with columns: user_prompt, result_truth, result_ai, retrieved_sources, retrieved_text.
    The output file name is the input CSV name with 'ai_result' appended before the extension if output_csv is not provided.
    """
    df = pd.read_csv(csv_path)
    outputs = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        user_prompt = row["user_prompt"]
        result_truth = row["result_truth"]
        
        # Get context and metadata
        context, metadata = retrieve_context(vectorstore, user_prompt, return_metadata=True)
        prompt = construct_prompt(config.SYSTEM_PROMPT, context, user_prompt)
        result_ai = generate_response(prompt)
        
        # Format sources and text for CSV
        sources = [meta.get('source', 'Unknown') for meta in metadata] if metadata else []
        sources_str = '; '.join(sources)
        context_str = '\n\n'.join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
        
        print(f"\nPrompt: {user_prompt}\n---\nRetrieved Context:\n{context_str}\n---\nAI Result:\n{result_ai}\n")
        outputs.append({
            "user_prompt": user_prompt, 
            "result_truth": result_truth, 
            "result_ai": result_ai,
            "retrieved_sources": sources_str,
            "retrieved_text": context_str
        })
    
    out_df = pd.DataFrame(outputs)
    if not output_csv:
        base, ext = os.path.splitext(csv_path)
        output_csv = f"{base}_ai_result{ext}"
    out_df.to_csv(output_csv, index=False)
    print(f"Batch evaluation complete. Output saved to {output_csv}") 