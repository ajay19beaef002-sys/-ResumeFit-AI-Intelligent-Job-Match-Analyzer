from typing import Optional
from io import BytesIO
import os
import PyPDF2
from docx import Document

def extract_text_from_pdf(path_or_bytes: str | bytes) -> str:
    """Extract raw text from a PDF file path or bytes."""
    try:
        if isinstance(path_or_bytes, bytes):
            reader = PyPDF2.PdfReader(BytesIO(path_or_bytes))
        else:
            reader = PyPDF2.PdfReader(path_or_bytes)
        texts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(texts)
    except Exception as e:
        raise ValueError(f"Failed to extract PDF: {str(e)}")

def extract_text_from_docx(path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        raise ValueError(f"Failed to extract DOCX: {str(e)}")

def extract_text_generic(path_or_bytes: str | bytes) -> str:
    """Extract text from PDF, DOCX, or TXT files."""
    try:
        if isinstance(path_or_bytes, bytes):
            # Assume PDF for bytes input
            return extract_text_from_pdf(path_or_bytes)
        ext = os.path.splitext(path_or_bytes)[1].lower()
        if ext == ".pdf":
            return extract_text_from_pdf(path_or_bytes)
        elif ext in (".docx", ".doc"):
            return extract_text_from_docx(path_or_bytes)
        else:
            with open(path_or_bytes, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        raise ValueError(f"Failed to extract text: {str(e)}")
