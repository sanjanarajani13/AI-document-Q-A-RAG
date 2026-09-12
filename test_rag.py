# test_rag.py

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import embed_chunks, get_model
from app.retrieval.vector_store import VectorStore
from app.generation.generator import generate_answer

pdf_path = "data/uploads/IMGT303L-POM - Module 5 _ FS 2026-27.pdf"  # update with your filename

# 1. Load and chunk
text = load_pdf(pdf_path)
chunks = chunk_text(text)

# 2. Embed and store
embeddings = embed_chunks(chunks)
store = VectorStore(dimension=embeddings.shape[1])
store.add(embeddings, chunks)

# 3. Ask a real question
question = input("Ask a question about your document: ")

query_embedding = get_model().encode(question)
relevant_chunks = store.search(query_embedding, top_k=3)

# 4. Generate an answer using Ollama
answer = generate_answer(question, relevant_chunks)

print("\n=== Answer ===")
print(answer)