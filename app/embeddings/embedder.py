# app/embeddings/embedder.py

from sentence_transformers import SentenceTransformer

# Small, fast, good-quality local embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_chunks(chunks: list[str]):
    """
    Converts a list of text chunks into embeddings.
    Returns a numpy array of shape (num_chunks, embedding_dim).
    """
    model = get_model()
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings