# test_embedding.py

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import embed_chunks, get_model
from app.retrieval.vector_store import VectorStore

pdf_path = "data/uploads/IMGT303L-POM - Module 5 _ FS 2026-27.pdf"  # update with your filename

# 1. Load and chunk
text = load_pdf(pdf_path)
chunks = chunk_text(text)
print(f"Created {len(chunks)} chunks.")

# 2. Embed
embeddings = embed_chunks(chunks)
print(f"Embeddings shape: {embeddings.shape}")

# 3. Store in FAISS
store = VectorStore(dimension=embeddings.shape[1])
store.add(embeddings, chunks)
print("Chunks added to FAISS index.")

# 4. Test a search query
query = "What is this document about?"  # try a real question about your PDF
query_embedding = get_model().encode(query)
results = store.search(query_embedding, top_k=3)

print("\nTop matching chunks for your query:\n")
for i, r in enumerate(results, 1):
    print(f"--- Result {i} ---\n{r}\n")