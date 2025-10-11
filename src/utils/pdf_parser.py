import pdfplumber

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts all text from a PDF file.
    Returns a single concatenated string.
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()
