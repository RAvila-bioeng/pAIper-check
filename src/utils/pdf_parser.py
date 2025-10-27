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
    # --- Fix common ligature issues ---
    ligatures = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        'ﬅ': 'ft', 'ﬆ': 'st'
    }
    for lig, repl in ligatures.items():
        text = text.replace(lig, repl)

    # --- Handle hyphenation ---
    # Fix broken words at line endings (e.g., "experi-\nment" -> "experiment")
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text, flags=re.IGNORECASE)

    # --- Normalize whitespace and line breaks ---
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Remove spaces around newlines
    text = re.sub(r'\s*\n\s*', '\n', text)
    # Normalize multiple newlines to a single paragraph break
    text = re.sub(r'\n{2,}', '\n\n', text)

    # --- Smartly join lines within paragraphs ---
    # This regex joins lines that end with a lowercase letter and are followed by another lowercase letter,
    # which is a good heuristic for sentence continuation.
    text = re.sub(r'(?<=[a-z,;])\n(?=[a-z])', ' ', text)

    # --- Remove artifacts ---
    # Remove page numbers (heuristic: a line with just numbers)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Remove headers/footers (heuristic: short lines appearing frequently)
    # This is complex; a simpler approach is to remove lines that don't end with punctuation
    # and are short, but this is risky. Let's stick to less destructive cleaning.

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
    
    # Improved patterns for section headers
    section_patterns = [
        # Numbered sections (more flexible): "1.", "1 ", "I.", "A."
        (r"^\s*([IVXLCDM]+\.|[A-Z]\.|\d{1,2}(?:\.\d{1,2})*\.?)\s+([A-Z][a-zA-Z0-9\s,:\-()]+)$", 'numbered'),
        
        # All caps headers (less restrictive) - AGREGADO: mínimo 4 caracteres
        (r"^\s*([A-Z][A-Z\s\-]{3,70})$", 'caps'),  # ← Cambiado de {5,70} a {3,70}
        
        # Title case headers (less restrictive)
        (r"^\s*([A-Z][a-z]+(?:\s+[A-Za-z]+){0,6})$", 'title_case'),  # ← Cambiado de {1,6} a {0,6}
    ]

    # Keywords que SI queremos detectar como headers
    priority_headers = {'references', 'bibliography', 'acknowledgments', 'appendix'}
    
    # Keywords that are unlikely to be section headers
    skip_keywords = {'abstract', 'conclusion'}  # Removidos: introduction, references

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # NUEVO: Detección especial para headers prioritarios
        line_lower = line.lower()
        if line_lower in priority_headers:
            pos = text.find(line)
            headers.append((pos, line, 'priority'))
            continue
            
        for pattern, header_type in section_patterns:
            match = re.match(pattern, line)
            if match:
                # Extract the full header text
                header_text = line
                
                # --- Filtering to avoid false positives ---
                # 1. Check length
                if len(header_text.split()) > 10 or len(header_text) > 150:
                    continue
                # 2. Check if it ends with punctuation (unlikely for a title)
                if header_text.endswith(('.', ':', ',')):
                    continue
                # 3. Check for common non-header keywords (case-insensitive)
                if header_text.lower() in skip_keywords:
                    # Allow them if they are part of a numbered list
                    if header_type != 'numbered' and not re.match(r"^\d", header_text):
                        continue
                
                # Find the position of this header in the full text
                try:
                    pos = text.find(header_text)
                    headers.append((pos, header_text, header_type))
                except ValueError:
                    pass
                
                # Once a line is matched as a header, stop checking other patterns
                break

    if not headers:
        return []

    # Sort by position and remove duplicates/overlaps
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
    
    
    # Manually search for "references" if not found by primary header detection
    if not any('reference' in h[1].lower() for h in headers):
        ref_match = re.search(r'(?:^|\n)\s*(references?|bibliography)\s*\n', text, re.IGNORECASE | re.MULTILINE)
        if ref_match:
            headers.append((ref_match.start(), ref_match.group(1).strip(), 'manual'))
            headers.sort(key=lambda x: x[0])
    
    # If header detection is weak, combine with fallback
    if len(headers) < 2:
        fallback_sections = extract_sections_fallback(text)
        if fallback_sections:
            return fallback_sections
        if not headers:  # No headers at all
            return []

    # Extract content between headers
    for i, (pos, header, htype) in enumerate(headers):
        start_pos = pos + len(header)
        end_pos = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        
        content = text[start_pos:end_pos].strip()
        
        # Further clean the content
        content = re.sub(r'\s+', ' ', content)
        
        # Ensure section is substantial before adding
        if len(content) > 200:
            # Clean header text by removing numbering/bullets
            clean_header = re.sub(r"^\s*([IVXLCDM]+\.|[A-Z]\.|\d{1,2}(?:\.\d{1,2})*\.?)\s*", '', header).strip()
            # SIN TRUNCAR EL CONTENIDO
            sections.append(Section(title=clean_header, content=content))
    
    # If the primary method yields very few sections, it might have failed.
    if len(sections) < 2:
        fallback_sections = extract_sections_fallback(text)
        # Simple merge: return fallback if it's better
        if len(fallback_sections) > len(sections):
            return fallback_sections

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
            # Create the keyword pattern string *before* the f-string
            all_keywords = sum(section_keywords.values(), [])
            cleaned_keywords = [p.replace("\\b", "").replace("\\", "") for p in all_keywords]
            keywords_pattern = "| ".join(cleaned_keywords)

            pattern = rf'(?:^|\n)\s*(?:\d+\.?\s*)?{pattern}\s*[:\-]?\s*\n+(.*?)(?=\n\s*(?:\d+\.?\s*)?(?:{keywords_pattern})|$)'
            
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            if match:
                content = match.group(1).strip()
                if len(content) > 100:
                    sections.append(Section(title=section_name, content=content[:5000]))
                    break
    
    return sections


def extract_references(text: str) -> List[Reference]:
    """Extract references with improved detection."""
    references = []
    
    # --- 1️⃣ Detectar bloque de referencias ---
    ref_patterns = [
        r'(?:^|\n)\s*references?\s*\n+(.*)',
        r'(?:^|\n)\s*bibliography\s*\n+(.*)',
        r'(?:^|\n)\s*(?:referencias|références)\s*\n+(.*)',
    ]

    ref_text = None
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if match:
            ref_text = match.group(1).strip()
            # Cortar anexos si los hay
            ref_text = re.split(
                r'\n\s*(appendix|acknowledgment|supplementary materials?)\b',
                ref_text,
                flags=re.IGNORECASE
            )[0]
            if len(ref_text) > 100:
                break

    if not ref_text:
        return references

    # --- 2️⃣ Dividir las referencias individuales ---
    # Intentar primero con patrones numerados
    ref_entries = re.split(
        r'\n\s*(?:\[\d+\]|\(\d+\)|\d+\.\s|\d+\)\s)',
        ref_text
    )

    # Si no hay muchas divisiones, probar heurísticas adicionales
    if len(ref_entries) < 3:
        ref_entries = re.split(r'\n{2,}|\n(?=[A-Z][a-z]+\s+[A-Z]\.)', ref_text)

    if len(ref_entries) < 5:
        ref_entries = re.split(r'\n\n+', ref_text)

    # --- 3️⃣ Procesar cada referencia ---
    for entry in ref_entries:
        entry = entry.strip()
        if len(entry) > 30:
            entry = re.sub(r'\s+', ' ', entry)
            doi_match = re.search(
                r'(?:doi[:\s]*|https?://doi\.org/)([^\s,;]+)',
                entry,
                re.IGNORECASE
            )
            doi = doi_match.group(1) if doi_match else None
            references.append(Reference(text=entry[:500], doi=doi))

    return references
