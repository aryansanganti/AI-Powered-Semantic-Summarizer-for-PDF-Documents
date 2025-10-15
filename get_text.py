import fitz  # PyMuPDF
import unicodedata

def extract_text_from_pdf(pdf_path):
    """Extract and normalize text from a PDF file."""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        # Normalize Unicode (fixes ligatures and special symbols)
        text = unicodedata.normalize("NFKC", text)
        return text.strip()
    except Exception as e:
        print(f"Failed to extract text from {pdf_path}: {e}")
        return ""
    finally:
        if doc is not None:
            doc.close()
