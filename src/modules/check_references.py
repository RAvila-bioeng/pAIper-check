"""
References and Citation Analysis Module.

This module evaluates the quality of the references section of a scientific paper based on a set of heuristics.
It focuses on:
- Quantity of references.
- Recency of sources.
- Consistency of formatting.
- Diversity of source types.
"""

import re
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter

from models.paper import Paper, Reference
from models.score import PillarResult

# -------------------- METRIC IMPLEMENTATIONS --------------------

def _check_reference_quantity(references: List[Reference]) -> Tuple[float, str]:
    """Scores the quantity of references."""
    count = len(references)
    if count < 10:
        score = 0.4
        feedback = f"⚠️ The paper has only {count} references, which may be insufficient for a scientific publication."
    elif 15 <= count <= 50:
        score = 1.0
        feedback = f"✅ The paper has a healthy number of references ({count})."
    elif count > 50:
        score = 0.8
        feedback = f"⚠️ The paper has a high number of references ({count}), which could suggest a lack of focus."
    else: # 10 <= count < 15
        score = 0.7
        feedback = f"✅ The number of references ({count}) is acceptable, though slightly low."
    return score, feedback

def _check_reference_recency(references: List[Reference]) -> Tuple[float, str]:
    """Scores the recency of references."""
    years = []
    for ref in references:
        # Find years between 1950 and the current year + 1
        found = re.findall(r'\b(19[5-9]\d|20\d{2})\b', ref.text)
        if found:
            years.append(int(found[0]))

    if not years:
        return 0.3, "⚠️ Could not determine the publication year for most references."

    current_year = datetime.now().year
    recent_count = sum(1 for y in years if y >= current_year - 10)
    ratio = recent_count / len(years)

    if ratio >= 0.7:
        score = 1.0
        feedback = f"✅ A high proportion ({int(ratio*100)}%) of references are from the last 10 years."
    elif ratio >= 0.4:
        score = 0.7
        feedback = f"✅ A reasonable proportion ({int(ratio*100)}%) of references are recent."
    else:
        score = 0.4
        feedback = f"⚠️ Many references seem outdated. Only {int(ratio*100)}% are from the last 10 years."
    return score, feedback

def _check_format_consistency(references: List[Reference]) -> Tuple[float, str]:
    """Scores the consistency of the reference format using heuristics."""
    patterns = []
    for ref in references:
        text = ref.text.strip()
        # Pattern 1: Starts with [1] or (1)
        if re.match(r'^[\(\[]\d+[\)\]]', text):
            patterns.append("numbered")
        # Pattern 2: Starts with an author name (e.g., "Smith, J.")
        elif re.match(r'^[A-Z][a-z]+,\s*[A-Z]\.', text):
            patterns.append("author_year")
        else:
            patterns.append("unknown")

    if not patterns:
        return 0.2, "⚠️ Could not identify a consistent citation format."

    # Find the most common pattern
    most_common = Counter(patterns).most_common(1)[0]
    dominant_pattern, count = most_common
    consistency_ratio = count / len(references)

    if dominant_pattern == "unknown":
        return 0.3, "⚠️ The reference format is highly inconsistent or uses a non-standard style."
    
    if consistency_ratio >= 0.85:
        score = 1.0
        feedback = "✅ The reference format appears to be consistent."
    elif consistency_ratio >= 0.6:
        score = 0.6
        feedback = "⚠️ The reference format shows some inconsistencies. Please verify all entries follow the same style."
    else:
        score = 0.3
        feedback = "❌ The reference format is highly inconsistent. A standard citation style should be applied uniformly."
    return score, feedback

def _check_source_diversity(references: List[Reference]) -> Tuple[float, str]:
    """Scores the diversity of reference sources."""
    source_types = []
    for ref in references:
        text = ref.text.lower()
        if any(keyword in text for keyword in ["journal", "transactions", "proceedings", "conference"]):
            source_types.append("journal/conference")
        elif "book" in text or "press" in text:
            source_types.append("book")
        elif "http" in text or "www" in text:
            source_types.append("web")
        else:
            source_types.append("other")
    
    counts = Counter(source_types)
    num_types = len(counts)

    if num_types >= 3:
        score = 1.0
        feedback = "✅ Good diversity of sources (journals, books, web, etc.)."
    elif num_types == 2:
        score = 0.8
        feedback = "✅ Source diversity is adequate, mainly consisting of two types."
    else:
        score = 0.5
        feedback = "⚠️ Source diversity could be improved. Most references are of the same type."

    # Penalize if web sources are dominant
    if len(references) > 0 and counts.get("web", 0) / len(references) > 0.5:
        score *= 0.7
        feedback += " However, there is a high reliance on web sources, which may not always be peer-reviewed."
        
    return score, feedback

# -------------------- MAIN EVALUATION LOGIC --------------------

def evaluate(paper: Paper, use_gpt: bool = False) -> dict:
    """
    Evaluates the quality of the paper's references.

    This function provides a baseline analysis of the references, which can be
    later enhanced by an LLM analysis.
    """
    references = paper.references

    if not references:
        feedback = "The paper does not contain a references section or it could not be parsed."
        result = PillarResult("References & Citations", 0.1, feedback).__dict__
        if use_gpt:
            result['gpt_analysis'] = {"used": False, "success": False, "error": "Skipped due to no references."}
        return result

    # 1. Evaluate each metric
    quantity_score, quantity_feedback = _check_reference_quantity(references)
    recency_score, recency_feedback = _check_reference_recency(references)
    consistency_score, consistency_feedback = _check_format_consistency(references)
    diversity_score, diversity_feedback = _check_source_diversity(references)

    # 2. Calculate weighted overall score
    weights = {
        'quantity': 0.25,
        'recency': 0.30,
        'consistency': 0.30,
        'diversity': 0.15
    }
    
    overall_score = (
        quantity_score * weights['quantity'] +
        recency_score * weights['recency'] +
        consistency_score * weights['consistency'] +
        diversity_score * weights['diversity']
    )
    overall_score = round(overall_score, 2)

    # 3. Generate combined feedback
    feedback = "\n  ".join([
        quantity_feedback,
        recency_feedback,
        consistency_feedback,
        diversity_feedback
    ])

    result = PillarResult("References & Citations", overall_score, feedback).__dict__

    # If the --use-chatgpt flag is used, activate the advanced analysis with Perplexity
    if use_gpt:
        try:
            from integrations.perplexity_api import analyze_references
            result['gpt_analysis'] = analyze_references(references)
        except ImportError:
            result['gpt_analysis'] = {
                "used": True,
                "success": False,
                "error": "Perplexity integration not found."
            }
        except Exception as e:
            result['gpt_analysis'] = {
                "used": True,
                "success": False,
                "error": f"Perplexity analysis failed: {e}"
            }
            
    return result
