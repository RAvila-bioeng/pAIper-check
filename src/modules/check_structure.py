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
    text = paper.full_text

    # Normalize for regex, keep original for slicing
    lowered_text = text.lower()

    # Canonical sections with common synonyms/variants
    section_map = {
        "abstract": [r"abstract", r"resumen", r"summary"],
        "introduction": [r"introduction", r"background", r"objetivos", r"introducción"],
        "methods": [r"methods?", r"methodology", r"materials?\s+and\s+methods"],
        "results": [r"results?", r"findings"],
        "discussion": [r"discussion", r"analysis"],
        "conclusion": [r"conclusions?", r"closing\s+remarks"],
        "references": [r"references", r"bibliography", r"bibliograf[íi]a", r"works\s+cited"],
    }

    # Build heading regex: start of line, optional numbering, optional colon/dot, heading word
    # e.g., "1. Introduction", "Introduction:", "Results", "2 Results"
    heading_patterns = {
        canonical: re.compile(
            rf"(?mi)^(?:\d+\s*[\.)]\s*)?(?:{'|'.join(variants)})\b\s*[:\.)-]?\s*$",
        )
        for canonical, variants in section_map.items()
    }

    # Find section positions (start indices) in text
    section_positions = {}
    for canonical, pattern in heading_patterns.items():
        matches = list(pattern.finditer(text))
        if matches:
            # take the first occurrence as the section start
            section_positions[canonical] = matches[0].start()

    # Determine found and missing from an expected core list
    expected_sections = [
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "references",
    ]
    found_sections = [s for s in expected_sections if s in section_positions]
    missing_sections = [s for s in expected_sections if s not in section_positions]

    # Completeness score favors core scientific article structure
    completeness_score = len(found_sections) / len(expected_sections)

    # Structure score based on sections object and per-section content length
    structure_score, short_sections = _check_section_structure(lowered_text, paper.sections)

    # Section order score: penalize if clearly out of order (when at least 3 are present)
    order_score, out_of_order = _check_section_order(section_positions, expected_sections)

    # Title quality
    title_score = _check_title_quality(paper.title)

    # Overall score: weighted average to reflect importance of completeness and order
    overall_score = min(
        1.0,
        (
            0.4 * completeness_score
            + 0.3 * structure_score
            + 0.2 * order_score
            + 0.1 * title_score
        ),
    )

    # Feedback
    feedback = _generate_structure_feedback(
        completeness_score=completeness_score,
        structure_score=structure_score,
        title_score=title_score,
        missing_sections=missing_sections,
        short_sections=short_sections,
        out_of_order=out_of_order,
    )

    return PillarResult("Structure & Completeness", overall_score, details={"missing": missing_sections, "short": short_sections, "out_of_order": out_of_order}
).__dict__


def _check_section_structure(text: str, sections: list) -> tuple:
    """Check if sections are properly structured.

    Returns a tuple (score, short_section_titles)
    """
    if not sections:
        return 0.3, []  # No sections detected

    score = 1.0

    # Minimum recommended content lengths per common section type
    minimum_lengths = {
        "abstract": 120,
        "introduction": 200,
        "methods": 300,
        "results": 250,
        "discussion": 250,
        "conclusion": 120,
        "references": 50,
    }

    short_sections = []
    for s in sections:
        title_key = s.title.lower().strip()
        # map to a known key heuristically
        mapped_key = None
        for key in minimum_lengths.keys():
            if key in title_key:
                mapped_key = key
                break
        min_len = minimum_lengths.get(mapped_key, 100)
        if len(s.content.strip()) < min_len:
            short_sections.append(s.title)

    if len(short_sections) > len(sections) * 0.3:
        score -= 0.3
    elif short_sections:
        score -= 0.15

    # Check for section numbering or clear headings
    if re.search(r"(?m)^\d+\s*[\.)]\s+\w+", text.lower()):
        score += 0.1

    return max(0.0, score), short_sections


def _check_section_order(section_positions: dict, expected_sections: list) -> tuple:
    """Compute order score and list of out-of-order sections."""
    present_positions = [(s, section_positions[s]) for s in expected_sections if s in section_positions]
    if len(present_positions) < 3:
        return 0.9, []  # not enough signal; be lenient

    # Extract positions and check monotonicity
    out_of_order = []
    last_pos = -1
    for s, pos in present_positions:
        if pos < last_pos:
            out_of_order.append(s)
        last_pos = max(last_pos, pos)

    if not out_of_order:
        return 1.0, []

    penalty = min(0.5, 0.1 * len(out_of_order))
    return max(0.0, 1.0 - penalty), out_of_order


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


def _generate_structure_feedback(
    completeness_score: float,
    structure_score: float,
    title_score: float,
    missing_sections: list,
    short_sections: list,
    out_of_order: list,
) -> str:
    """Generate detailed feedback for structure evaluation."""
    feedback_parts = []

    if missing_sections:
        feedback_parts.append(
            f"Missing essential sections: {', '.join(missing_sections)}."
        )

    if out_of_order:
        feedback_parts.append(
            f"Sections appear out of order: {', '.join(out_of_order)}. Consider standard IMRaD flow."
        )

    if short_sections:
        feedback_parts.append(
            f"Some sections seem too brief: {', '.join(short_sections)}. Expand with essential details."
        )

    if structure_score < 0.7 and not short_sections:
        feedback_parts.append(
            "Section structure could be improved with better organization and content distribution."
        )

    if title_score < 0.7:
        feedback_parts.append(
            "Title quality needs improvement. Ensure proper capitalization and appropriate length."
        )

    if not feedback_parts:
        feedback_parts.append(
            "Good structure with essential sections present, in order, and well-organized."
        )

    return " ".join(feedback_parts)
