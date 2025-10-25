import pdfplumber
import re
from pathlib import Path
from typing import Dict, List, Optional
from models.paper import Paper, Section, Reference


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


def parse_paper_from_pdf(file_path: str) -> Paper:
    """
    Parse a PDF or text file and extract structured information.
    Returns a Paper object with sections and metadata.
    """
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        raw_text = extract_text_from_pdf(file_path)
    elif file_extension in ['.txt', '.text']:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Extract title (usually the first non-empty line)
    title = extract_title(raw_text)
    
    # Extract abstract
    abstract = extract_abstract(raw_text)
    
    # Extract sections
    sections = extract_sections(raw_text)
    
    # Extract references
    references = extract_references(raw_text)
    
    return Paper(
        raw_text=raw_text,
        title=title,
        abstract=abstract,
        sections=sections,
        references=references
    )


def extract_title(text: str) -> str:
    """Extract the title from the paper text."""
    lines = text.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if len(line) > 10 and len(line) < 200:  # Reasonable title length
            # Skip common headers
            if not any(header in line.lower() for header in ['abstract', 'introduction', 'method', 'result']):
                return line
    return ""


def extract_abstract(text: str) -> str:
    """Extract the abstract section."""
    # Look for abstract section
    abstract_patterns = [
        r'abstract\s*[:\-]?\s*(.*?)(?=keywords|introduction|1\.|i\.|$)',
        r'resumen\s*[:\-]?\s*(.*?)(?=palabras clave|introducción|1\.|i\.|$)',
        r'abstract\s*[:\-]?\s*(.*?)(?=introduction|1\.|i\.|$)',
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            # Clean up the abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract[:1000]  # Limit abstract length
    
    return ""


def extract_sections(text: str) -> List[Section]:
    """Extract main sections from the paper."""
    sections = []
    
    # Common section patterns
    section_patterns = [
        r'(introduction|introducción)\s*[:\-]?\s*(.*?)(?=method|methodology|método|metodología|result|resultado|conclusion|conclusión|discussion|discusión|$)',
        r'(methodology|methods|método|metodología)\s*[:\-]?\s*(.*?)(?=result|resultado|conclusion|conclusión|discussion|discusión|$)',
        r'(results|resultados)\s*[:\-]?\s*(.*?)(?=conclusion|conclusión|discussion|discusión|$)',
        r'(discussion|discusión)\s*[:\-]?\s*(.*?)(?=conclusion|conclusión|$)',
        r'(conclusion|conclusiones|conclusión|conclusiones)\s*[:\-]?\s*(.*?)(?=references|referencias|bibliography|bibliografía|$)',
    ]
    
    for pattern in section_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            section_title = match.group(1).strip()
            section_content = match.group(2).strip()
            # Clean up content
            section_content = re.sub(r'\s+', ' ', section_content)
            if len(section_content) > 50:  # Only add substantial sections
                sections.append(Section(title=section_title, content=section_content))
    
    return sections


def extract_references(text: str) -> List[Reference]:
    """Extract references from the paper."""
    references = []
    
    # Look for references section
    ref_patterns = [
        r'(references|referencias|bibliography|bibliografía)\s*[:\-]?\s*(.*?)$',
        r'(references|referencias|bibliography|bibliografía)\s*[:\-]?\s*(.*?)(?=\n\n|\n\s*\n)',
    ]
    
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            ref_text = match.group(2).strip()
            # Split references by common patterns
            ref_entries = re.split(r'\n(?=\d+\.|\d+\)|\[)', ref_text)
            
            for entry in ref_entries:
                entry = entry.strip()
                if len(entry) > 20:  # Only substantial references
                    # Try to extract DOI if present
                    doi_match = re.search(r'doi[:\s]*([^\s,]+)', entry, re.IGNORECASE)
                    doi = doi_match.group(1) if doi_match else None
                    
                    references.append(Reference(text=entry, doi=doi))
            break
    
    return references
