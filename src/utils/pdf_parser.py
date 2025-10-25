import re
from pathlib import Path
from typing import List, Optional
from PyPDF2 import PdfReader
from models.paper import Paper, Section, Reference


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts all text from a PDF file using PyPDF2.
    Returns a single concatenated string.
    """
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF with PyPDF2: {e}")
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except ImportError:
            print("pdfplumber is not installed. Cannot fallback.")
        except Exception as pl_e:
            print(f"Fallback to pdfplumber also failed: {pl_e}")
    
    return text.strip()


def clean_text(text: str) -> str:
    """
    Clean extracted text by fixing common PDF extraction issues.
    """
    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r' +', ' ', text)
    
    # Fix broken words at line endings (hyphenation)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Fix line breaks within sentences (but keep paragraph breaks)
    text = re.sub(r'(?<![.!?:])\n(?=[a-z])', ' ', text)
    
    # Normalize multiple newlines to double newline (paragraph break)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    return text


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
    
    # Clean the text
    cleaned_text = clean_text(raw_text)
    
    # Extract title
    title = extract_title(cleaned_text)
    
    # Extract abstract
    abstract = extract_abstract(cleaned_text)
    
    # Extract sections
    sections = extract_sections(cleaned_text)
    
    # Extract references
    references = extract_references(cleaned_text)
    
    return Paper(
        raw_text=cleaned_text,
        title=title,
        abstract=abstract,
        sections=sections,
        references=references
    )


def extract_title(text: str) -> str:
    """Extract the title from the paper text."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Skip common headers that appear before title
    skip_keywords = ['journal', 'volume', 'article', 'doi', 'received', 'accepted']
    
    for i, line in enumerate(lines[:20]):
        # Skip very short lines or lines with skip keywords
        if len(line) < 15 or any(kw in line.lower() for kw in skip_keywords):
            continue
        
        # Skip lines that look like author names (contain common name patterns)
        if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', line) and len(line) < 50:
            continue
        
        # Skip lines with emails or affiliations
        if '@' in line or re.search(r'\d{5}', line):
            continue
        
        # Title should be substantial and not all caps (unless it's a proper title)
        if 20 <= len(line) <= 250:
            return line
    
    return lines[0] if lines else ""


def extract_abstract(text: str) -> str:
    """Extract the abstract section with improved pattern matching."""
    # Multiple patterns to catch different abstract formats
    patterns = [
        # Standard format with "abstract" header
        r'(?:^|\n)\s*abstract\s*\n+(.*?)(?=\n\s*(?:keywords?|introduction|1\.?\s+introduction|\d+\.?\s+\w+|$))',
        
        # Abstract with colon or dash
        r'(?:^|\n)\s*abstract\s*[:\-]\s*(.*?)(?=\n\s*(?:keywords?|introduction|1\.?\s+introduction|$))',
        
        # Abstract in other languages
        r'(?:^|\n)\s*(?:resumen|résumé)\s*[:\-]?\s*(.*?)(?=\n\s*(?:keywords?|palabras clave|introduction|$))',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if match:
            abstract = match.group(1).strip()
            # Clean up the abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            # Remove common artifacts
            abstract = re.sub(r'\b(?:keywords?|introduction)\b.*$', '', abstract, flags=re.IGNORECASE)
            
            if len(abstract) > 50:  # Ensure it's substantial
                return abstract[:2000]
    
    return ""


def find_section_headers(text: str) -> List[tuple]:
    """
    Find all section headers in the text with their positions.
    Returns list of (position, header_text, header_type) tuples.
    """
    headers = []
    
    # Common section patterns in scientific papers
    section_patterns = [
        # Numbered sections: "1. Introduction" or "1 Introduction"
        (r'\n\s*(\d+\.?\s+[A-Z][a-zA-Z\s]+)\s*\n', 'numbered'),
        
        # All caps headers: "INTRODUCTION"
        (r'\n\s*([A-Z][A-Z\s]{5,50})\s*\n', 'caps'),
        
        # Standard headers: "Introduction"
        (r'\n\s*([A-Z][a-z]+(?:\s+(?:and|&|of|for|the)\s+[A-Z][a-z]+)*)\s*\n', 'standard'),
    ]
    
    for pattern, header_type in section_patterns:
        for match in re.finditer(pattern, text):
            header_text = match.group(1).strip()
            
            # Filter out false positives
            if len(header_text.split()) > 8:  # Too long for a header
                continue
            if header_text.lower() in ['a b s t r a c t', 'the', 'this', 'these']:
                continue
                
            headers.append((match.start(), header_text, header_type))
    
    # Sort by position and remove duplicates
    headers.sort(key=lambda x: x[0])
    
    # Remove overlapping headers (keep the first one)
    cleaned_headers = []
    last_pos = -100
    for pos, header, htype in headers:
        if pos - last_pos > 50:  # Minimum distance between headers
            cleaned_headers.append((pos, header, htype))
            last_pos = pos
    
    return cleaned_headers


def extract_sections(text: str) -> List[Section]:
    """
    Extract main sections from the paper using improved detection.
    """
    sections = []
    
    # Find all section headers
    headers = find_section_headers(text)
    
    if not headers:
        # Fallback to basic extraction if no headers found
        return extract_sections_fallback(text)
    
    # Extract content between headers
    for i, (pos, header, htype) in enumerate(headers):
        # Determine end position (next header or end of text)
        end_pos = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        
        # Extract section content
        content = text[pos:end_pos].strip()
        
        # Remove the header from content
        content = content[len(header):].strip()
        
        # Clean up content
        content = re.sub(r'\s+', ' ', content)
        
        # Only add substantial sections
        if len(content) > 100:
            # Clean header
            clean_header = re.sub(r'^\d+\.?\s*', '', header).strip()
            sections.append(Section(title=clean_header, content=content[:5000]))
    
    # If we found very few sections, try fallback method
    if len(sections) < 3:
        return extract_sections_fallback(text)
    
    return sections


def extract_sections_fallback(text: str) -> List[Section]:
    """
    Fallback method for section extraction using keyword matching.
    """
    sections = []
    
    # Key sections to look for
    section_keywords = {
        'Introduction': [r'\bintroduction\b', r'\bbackground\b'],
        'Methods': [r'\bmethods?\b', r'\bmethodology\b', r'\bmaterials?\s+and\s+methods?\b', r'\bexperimental\b'],
        'Results': [r'\bresults?\b', r'\bfindings?\b'],
        'Discussion': [r'\bdiscussion\b', r'\banalysis\b'],
        'Conclusion': [r'\bconclusions?\b', r'\bsummary\b'],
    }
    
    for section_name, patterns in section_keywords.items():
        for pattern in patterns:
            # Look for section with flexible matching
            regex = rf'(?:^|\n)\s*(?:\d+\.?\s*)?{pattern}\s*[:\-]?\s*\n+(.*?)(?=\n\s*(?:\d+\.?\s*)?(?:{"| ".join([p.replace("\\b", "").replace("\\", "") for p in sum(section_keywords.values(), [])])})|$)'
            
            match = re.search(regex, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            if match:
                content = match.group(1).strip()
                content = re.sub(r'\s+', ' ', content)
                
                if len(content) > 100:
                    sections.append(Section(title=section_name, content=content[:5000]))
                    break  # Found this section, move to next
    
    return sections


def extract_references(text: str) -> List[Reference]:
    """Extract references with improved detection."""
    references = []
    
    # Find references section
    ref_patterns = [
        r'(?:^|\n)\s*references?\s*\n+(.*?)$',
        r'(?:^|\n)\s*bibliography\s*\n+(.*?)$',
        r'(?:^|\n)\s*(?:referencias|références)\s*\n+(.*?)$',
    ]
    
    ref_text = None
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if match:
            ref_text = match.group(1).strip()
            break
    
    if not ref_text:
        return references
    
    # Split references by common patterns
    # Look for: [1], [1], (1), 1., or 1)
    ref_entries = re.split(r'\n\s*(?:\[\d+\]|\(\d+\)|\d+\.|\d+\))\s+', ref_text)
    
    for entry in ref_entries:
        entry = entry.strip()
        if len(entry) > 30:  # Only substantial references
            # Clean up the entry
            entry = re.sub(r'\s+', ' ', entry)
            
            # Try to extract DOI
            doi_match = re.search(r'(?:doi[:\s]*|https?://doi\.org/)([^\s,;]+)', entry, re.IGNORECASE)
            doi = doi_match.group(1) if doi_match else None
            
            references.append(Reference(text=entry[:500], doi=doi))
    
    return references