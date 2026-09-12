# app/ingestion/pdf_loader.py

from pypdf import PdfReader

def load_pdf(file_path: str) -> str:
    """
    Extracts all text from a PDF file and returns it as a single string.
    """
    reader = PdfReader(file_path)
    full_text = ""

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    return full_text