import re
from models.score import PillarResult

def evaluate(paper):
    """
    Evaluate structure and completeness of the paper.
    
    Args:
        paper: Paper object with text content
        
    Returns:
        dict: Score and feedback for structure evaluation
    """
    text = paper.full_text.lower()
    
    # Check for essential sections
    essential_sections = ["abstract", "introduction", "method", "results", "conclusion", "references"]
    found_sections = []
    missing_sections = []
    
    for section in essential_sections:
        if section in text:
            found_sections.append(section)
        else:
            missing_sections.append(section)
    
    # Calculate completeness score
    completeness_score = len(found_sections) / len(essential_sections)
    
    # Check for proper section structure
    structure_score = _check_section_structure(text, paper.sections)
    
    # Check for title quality
    title_score = _check_title_quality(paper.title)
    
    # Calculate overall score
    overall_score = min(1.0, (completeness_score + structure_score + title_score) / 3)
    
    # Generate feedback
    feedback = _generate_structure_feedback(completeness_score, structure_score, title_score, missing_sections)
    
    return PillarResult("Structure & Completeness", overall_score, feedback).__dict__


def _check_section_structure(text: str, sections: list) -> float:
    """Check if sections are properly structured."""
    if not sections:
        return 0.3  # No sections detected
    
    score = 1.0
    
    # Check if sections have reasonable content length
    short_sections = [s for s in sections if len(s.content) < 100]
    if len(short_sections) > len(sections) * 0.3:
        score -= 0.3
    
    # Check for section numbering or clear headings
    numbered_sections = len(re.findall(r'\d+\.\s+\w+', text))
    if numbered_sections > 0:
        score += 0.1
    
    return max(0.0, score)


def _check_title_quality(title: str) -> float:
    """Check the quality of the paper title."""
    if not title:
        return 0.2
    
    score = 1.0
    
    # Check title length (should be reasonable)
    if len(title) < 10:
        score -= 0.4
    elif len(title) > 200:
        score -= 0.2
    
    # Check for appropriate capitalization
    words = title.split()
    if words:
        # First word should be capitalized
        if not words[0][0].isupper():
            score -= 0.2
        
        # Other major words should be capitalized (title case)
        major_words = [w for w in words[1:] if len(w) > 3 and not w.lower() in ['and', 'or', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by']]
        uncapitalized_major = [w for w in major_words if not w[0].isupper()]
        if uncapitalized_major:
            score -= 0.1
    
    return max(0.0, score)


def _generate_structure_feedback(completeness_score: float, structure_score: float, 
                               title_score: float, missing_sections: list) -> str:
    """Generate detailed feedback for structure evaluation."""
    feedback_parts = []
    
    if completeness_score < 0.8:
        if missing_sections:
            feedback_parts.append(f"Missing essential sections: {', '.join(missing_sections)}.")
    
    if structure_score < 0.7:
        feedback_parts.append("Section structure could be improved with better organization and content distribution.")
    
    if title_score < 0.7:
        feedback_parts.append("Title quality needs improvement. Ensure proper capitalization and appropriate length.")
    
    if not feedback_parts:
        feedback_parts.append("Good structure with all essential sections present and well-organized.")
    
    return " ".join(feedback_parts)
