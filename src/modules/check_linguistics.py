"""
Linguistic Quality Control Module
Focuses on precision and style of scientific language.
"""

import re
from typing import List, Tuple
from models.score import PillarResult


def evaluate(paper) -> dict:
    """
    Evaluate linguistic quality of the paper.
    
    Args:
        paper: Paper object with text content
        
    Returns:
        dict: Score and feedback for linguistic quality
    """
    text = paper.full_text.lower()
    
    # Check for spelling errors (basic patterns)
    spelling_score = _check_spelling(text)
    
    # Check terminology consistency
    terminology_score = _check_terminology_consistency(paper)
    
    # Check academic style
    style_score = _check_academic_style(text)
    
    # Check grammar patterns
    grammar_score = _check_grammar_patterns(text)
    
    # Calculate overall score
    overall_score = (spelling_score + terminology_score + style_score + grammar_score) / 4
    
    # Generate feedback
    feedback = _generate_linguistic_feedback(spelling_score, terminology_score, style_score, grammar_score)
    
    return PillarResult("Linguistic Quality", overall_score, feedback).__dict__


def _check_spelling(text: str) -> float:
    """Check for common spelling errors and typos."""
    # Common misspellings in academic writing
    misspellings = [
        r'\bteh\b',  # the
        r'\badn\b',  # and
        r'\bhte\b',  # the
        r'\btaht\b', # that
        r'\bwih\b',  # with
        r'\bthier\b', # their
        r'\brecieve\b', # receive
        r'\boccured\b', # occurred
        r'\bseperate\b', # separate
        r'\bdefinately\b', # definitely
    ]
    
    error_count = 0
    for pattern in misspellings:
        if re.search(pattern, text):
            error_count += 1
    
    # Score decreases with more errors
    if error_count == 0:
        return 1.0
    elif error_count <= 2:
        return 0.8
    elif error_count <= 5:
        return 0.6
    else:
        return 0.3


def _check_terminology_consistency(paper) -> float:
    """Check for consistent use of technical terms."""
    text = paper.full_text.lower()
    
    # Common scientific terms that should be consistent
    term_pairs = [
        ('data set', 'dataset'),
        ('data-set', 'dataset'),
        ('machine learning', 'ml'),
        ('artificial intelligence', 'ai'),
        ('neural network', 'nn'),
        ('deep learning', 'dl'),
    ]
    
    inconsistency_count = 0
    for term1, term2 in term_pairs:
        count1 = len(re.findall(rf'\b{re.escape(term1)}\b', text))
        count2 = len(re.findall(rf'\b{re.escape(term2)}\b', text))
        
        if count1 > 0 and count2 > 0:
            inconsistency_count += 1
    
    if inconsistency_count == 0:
        return 1.0
    elif inconsistency_count <= 2:
        return 0.8
    else:
        return 0.6


def _check_academic_style(text: str) -> float:
    """Check for appropriate academic writing style."""
    score = 1.0
    
    # Check for informal language
    informal_patterns = [
        r'\bawesome\b', r'\bcool\b', r'\bnice\b', r'\bgreat\b',
        r'\blots of\b', r'\btons of\b', r'\bpretty much\b',
        r'\bkind of\b', r'\bsort of\b', r'\bvery very\b'
    ]
    
    informal_count = sum(len(re.findall(pattern, text)) for pattern in informal_patterns)
    if informal_count > 0:
        score -= min(0.3, informal_count * 0.05)
    
    # Check for appropriate use of passive voice (common in academic writing)
    passive_patterns = [
        r'\bis\s+\w+ed\b', r'\bwas\s+\w+ed\b', r'\bwere\s+\w+ed\b',
        r'\bhas\s+been\s+\w+ed\b', r'\bhave\s+been\s+\w+ed\b'
    ]
    
    passive_count = sum(len(re.findall(pattern, text)) for pattern in passive_patterns)
    if passive_count < 5:  # Too few passive constructions
        score -= 0.1
    
    # Check for appropriate use of citations
    citation_patterns = [
        r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
        r'\[[^\]]*\d{4}[^\]]*\]',  # [Author, 2023]
    ]
    
    citation_count = sum(len(re.findall(pattern, text)) for pattern in citation_patterns)
    if citation_count < 5:  # Too few citations
        score -= 0.2
    
    return max(0.0, score)


def _check_grammar_patterns(text: str) -> float:
    """Check for common grammar issues."""
    score = 1.0
    
    # Check for sentence fragments (very basic)
    sentences = re.split(r'[.!?]+', text)
    fragment_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Very short sentences might be fragments
            fragment_count += 1
    
    if fragment_count > len(sentences) * 0.1:  # More than 10% fragments
        score -= 0.2
    
    # Check for run-on sentences (very basic)
    long_sentences = [s for s in sentences if len(s) > 200]
    if len(long_sentences) > len(sentences) * 0.2:  # More than 20% very long sentences
        score -= 0.2
    
    # Check for proper capitalization after periods
    capitalization_errors = len(re.findall(r'\.\s+[a-z]', text))
    if capitalization_errors > 0:
        score -= min(0.3, capitalization_errors * 0.05)
    
    return max(0.0, score)


def _generate_linguistic_feedback(spelling_score: float, terminology_score: float, 
                                 style_score: float, grammar_score: float) -> str:
    """Generate detailed feedback for linguistic quality."""
    feedback_parts = []
    
    if spelling_score < 0.8:
        feedback_parts.append("Some spelling errors detected. Consider using a spell checker.")
    
    if terminology_score < 0.8:
        feedback_parts.append("Inconsistent use of technical terminology. Ensure consistent terminology throughout.")
    
    if style_score < 0.8:
        feedback_parts.append("Some informal language detected. Maintain formal academic tone throughout.")
    
    if grammar_score < 0.8:
        feedback_parts.append("Some grammar issues detected. Review sentence structure and punctuation.")
    
    if not feedback_parts:
        feedback_parts.append("Good linguistic quality with appropriate academic style and terminology.")
    
    return " ".join(feedback_parts)
