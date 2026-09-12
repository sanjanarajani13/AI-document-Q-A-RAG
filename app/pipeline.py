from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import embed_chunks, get_model
from app.retrieval.vector_store import VectorStore
from app.generation.generator import generate_answer

def process_pdf(file_path: str) -> VectorStore:
    text = load_pdf(file_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    store = VectorStore(dimension=embeddings.shape[1])
    store.add(embeddings, chunks)
    return store

def answer_question(store: VectorStore, question: str, top_k: int = 3) -> str:
    query_embedding = get_model().encode(question)
    relevant_chunks = store.search(query_embedding, top_k=top_k)
    return generate_answer(question, relevant_chunks)