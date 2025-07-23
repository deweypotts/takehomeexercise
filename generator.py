"""
Prompt construction and OpenAI generation logic for RAG pipeline.
"""
from typing import List
import openai
import config

def construct_prompt(system_prompt: str, context: List[str], user_prompt: str) -> List[dict]:
    """
    Constructs the prompt for OpenAI Chat Completion API.
    """
    context_str = "\n\n".join(context)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_prompt}"}
    ]

def generate_response(prompt: List[dict], model: str = None) -> str:
    """
    Calls OpenAI's Chat Completion API and returns the response.
    """
    if model is None:
        model = config.CHAT_MODEL
    response = openai.ChatCompletion.create(
        model=model,
        messages=prompt,
        temperature=0.2,
        max_tokens=512
    )
    return response.choices[0].message["content"].strip() 