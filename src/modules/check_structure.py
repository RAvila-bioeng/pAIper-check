import re
from models.score import PillarResult

def evaluate(paper, use_gpt=False):
    """
    Evaluate structure and completeness of the paper based on pre-parsed sections.
    
    Args:
        paper: Paper object with a list of Section objects
        use_gpt (bool): Flag to enable Perplexity analysis.
        
    Returns:
        dict: Score and feedback for structure evaluation
    """
    # Canonical sections and their primary, accepted variants
    section_map = {
        "abstract": [r"abstract", r"summary"],
        "introduction": [r"introduction", r"background"],
        "methods": [r"methods?", r"methodology", r"materials?\s+and\s+methods", r"experimental\s+setup"],
        "results": [r"results?", r"findings", r"experiments", r"experimental\s+results"],
        "discussion": [r"discussion", r"analysis"],
        "conclusion": [r"conclusions?", r"concluding\s+remarks", r"summary\s+and\s+conclusion"],
        "references": [r"references", r"bibliography", r"works\s+cited"],
    }

    # Aliases that are valid but should trigger a suggestion
    section_aliases = {
        "abstract": [r"resumen"],
        "introduction": [r"objetivos", r"introducción"],
        "conclusion": [r"discussion", r"final\s+remarks"], # Allow discussion as a conclusion
        "references": [r"bibliograf[íi]a"],
    }
    
    # --- Refactored Logic: Use paper.sections ---
    
    # 1. Map parsed section titles to canonical names
    found_canonical_sections = []
    used_aliases = {}
    
    all_variants = {**section_map}
    for key, aliases in section_aliases.items():
        all_variants[key] = all_variants.get(key, []) + aliases

    for section in paper.sections:
        title_lower = section.title.lower().strip()
        matched = False
        for canonical, variants in all_variants.items():
            for variant_pattern in variants:
                if re.search(r'\b' + variant_pattern + r'\b', title_lower):
                    if canonical not in found_canonical_sections: # Add only first match
                        found_canonical_sections.append(canonical)
                    
                    # Check if an alias was used
                    if canonical in section_aliases and variant_pattern in section_aliases[canonical]:
                        used_aliases[canonical] = section.title.strip()

                    matched = True
                    break
            if matched:
                break

    # 2. Determine completeness
    expected_sections = [
        "abstract", "introduction", "methods", "results", 
        "discussion", "conclusion", "references",
    ]
    
    found_set = set(found_canonical_sections)
    missing_sections = [s for s in expected_sections if s not in found_set]
    completeness_score = len(found_set) / len(expected_sections)

    # 3. Check structure (content length) - reuses existing helper
    structure_score, short_sections = _check_section_structure(paper.full_text, paper.sections)

    # 4. Check section order
    order_score, out_of_order = _check_section_order_refactored(found_canonical_sections, expected_sections)

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
        used_aliases=used_aliases,
    )

    result = PillarResult(
        pillar_name="Structure & Completeness",
        score=overall_score,
        feedback=feedback,
    ).__dict__

    # 🔹 Si se usa la opción --use-chatGPT, activar el análisis avanzado con Perplexity
    if use_gpt:
        try:
            from integrations.perplexity_api import analyze_structure
            # El análisis se basa en la lista de secciones del paper
            result['gpt_analysis'] = analyze_structure(paper.sections)
        except ImportError:
            result['gpt_analysis'] = {
                "success": False,
                "error": "Perplexity integration not found."
            }
        except Exception as e:
            result['gpt_analysis'] = {
                "success": False,
                "error": f"Perplexity analysis failed: {e}"
            }
            
    return result


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


def _check_section_order_refactored(found_sections: list, expected_order: list) -> tuple:
    """
    Compute order score based on the sequence of found canonical sections.
    """
    if len(found_sections) < 3:
        return 0.9, []  # Not enough sections to reliably determine order

    # Get the indices of found sections from the expected order
    indices = [expected_order.index(s) for s in found_sections if s in expected_order]
    
    # Check if the indices are monotonically increasing
    out_of_order = []
    for i in range(len(indices) - 1):
        if indices[i] > indices[i+1]:
            # The section at found_sections[i+1] is out of order
            out_of_order.append(found_sections[i+1])

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
    used_aliases: dict,
) -> str:
    """Generate detailed and balanced feedback for structure evaluation."""
    feedback_parts = []

    # 1. Completeness Feedback
    if completeness_score == 1.0:
        feedback_parts.append("✓ Excellent completeness: All standard sections (Abstract, Introduction, Methods, etc.) are present.")
    elif completeness_score >= 0.7:
        feedback_parts.append(f"✓ Good completeness, although some standard sections are missing: {', '.join(missing_sections)}.")
    else:
        feedback_parts.append(f"⚠️ Completeness needs improvement. Key sections are missing: {', '.join(missing_sections)}.")

    # 2. Section Order Feedback
    if not out_of_order:
        feedback_parts.append("✓ The sections follow a logical and conventional order (IMRaD).")
    else:
        feedback_parts.append(f"⚠️ The section order could be improved. Sections found out of order: {', '.join(out_of_order)}. Consider the standard IMRaD flow.")

    # 3. Content Length Feedback
    if not short_sections:
        feedback_parts.append("✓ All sections appear to have sufficient content.")
    else:
        feedback_parts.append(f"⚠️ Some sections seem too brief: {', '.join(short_sections)}. Ensure they contain enough detail for the reader.")

    # 4. Title Quality Feedback
    if title_score >= 0.8:
        feedback_parts.append("✓ The paper has a well-formatted and appropriately long title.")
    else:
        feedback_parts.append("⚠️ The title could be improved. Ensure it is descriptive, not too long, and follows standard capitalization rules.")

    # 5. Use of Aliases (informational, not penalizing)
    if used_aliases:
        alias_feedback = [f"'{alias}' (for '{canonical}')" for canonical, alias in used_aliases.items()]
        feedback_parts.append(f"• Informational: Non-standard section titles were used: {'; '.join(alias_feedback)}. Consider using standard English terms for broader clarity.")

    return "\n  ".join(feedback_parts)
