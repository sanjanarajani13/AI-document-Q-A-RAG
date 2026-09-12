# test_ingestion.py

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_text

# Change this to match your actual PDF filename
pdf_path = "data/uploads/IMGT303L-POM - Module 5 _ FS 2026-27.pdf"

text = load_pdf(pdf_path)
print(f"Extracted {len(text)} characters.\n")

chunks = chunk_text(text)
print(f"Split into {len(chunks)} chunks.\n")

print("First chunk preview:\n")
print(chunks[0])