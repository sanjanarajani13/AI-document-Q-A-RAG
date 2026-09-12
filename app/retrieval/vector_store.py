# app/retrieval/vector_store.py

import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []  # keeps text aligned with vector positions

    def add(self, embeddings: np.ndarray, chunks: list[str]):
        self.index.add(np.array(embeddings).astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), top_k
        )
        results = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return results

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, folder: str):
        self.index = faiss.read_index(os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)