# app/generation/generator.py

import ollama

MODEL_NAME = "llama3.2"

def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant answering questions based ONLY on the context provided below.
If the answer is not contained in the context, say "I don't have enough information in the document to answer that."

Context:
{context}

Question: {question}

Answer:"""
    return prompt

def generate_answer(question: str, context_chunks: list[str]) -> str:
    prompt = build_prompt(question, context_chunks)
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]