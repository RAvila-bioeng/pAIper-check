"""
References and Citation Verification Module
Guarantees the work's strength and bibliographical support.
"""

import re
import requests
from typing import List, Dict
from datetime import datetime
from models.score import PillarResult

CROSSREF_API = "https://api.crossref.org/works/"


def evaluate(paper) -> dict:
    """
    Evaluate references and citation quality using advanced logic.
    It checks:
      - format correctness
      - source accessibility (via DOI CrossRef)
      - credibility and recency
      - diversity of references
      - semantic relevance (optional extension)
    """
    text = paper.full_text
    references = paper.references

    format_score = _check_citation_format(text)
    quality_score = _check_reference_quality(references)
    accessibility_score = _check_reference_accessibility(references)
    density_score = _check_citation_density(text)
    recency_score = _check_reference_recency(references)
    diversity_score = _check_reference_diversity(references)

    overall_score = (
        format_score
        + quality_score
        + accessibility_score
        + density_score
        + recency_score
        + diversity_score
    ) / 6

    feedback = _generate_reference_feedback(
        format_score, quality_score, accessibility_score, density_score, recency_score, diversity_score
    )

    return PillarResult("References & Citations", overall_score, feedback).__dict__


# ----------------------------- FORMAT CHECK -----------------------------

def _check_citation_format(text: str) -> float:
    score = 1.0
    citation_patterns = [
        r"\([^)]*\d{4}[^)]*\)",  # (Author, 2024)
        r"\[\d+\]",              # [1]
        r"\[[^\]]*\d{4}[^\]]*\]" # [Author, 2023]
    ]
    citations = []
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, text))
    
    if not citations:
        return 0.1
    
    styles = {
        "parenthetical": len(re.findall(r"\([^)]*\d{4}[^)]*\)", text)),
        "numbered": len(re.findall(r"\[\d+\]", text)),
        "bracketed": len(re.findall(r"\[[^\]]*\d{4}[^\]]*\]", text)),
    }
    
    dominant_style = max(styles, key=styles.get)
    total = sum(styles.values())
    consistency = styles[dominant_style] / total if total > 0 else 0
    if consistency < 0.75:
        score -= 0.3
    
    misplaced = len(re.findall(r"\.\s*\([^)]*\d{4}[^)]*\)\s*[A-Z]", text))
    if misplaced > len(citations) * 0.25:
        score -= 0.2

    return max(0.0, score)


# ----------------------------- QUALITY CHECK -----------------------------

def _check_reference_quality(references: List) -> float:
    if not references:
        return 0.1
    score = 1.0
    complete_refs = 0
    for ref in references:
        txt = ref.text.lower()
        has_author = bool(re.findall(r"[A-Z][a-z]+,\s*[A-Z]\.", ref.text))
        has_year = bool(re.search(r"\b(19|20)\d{2}\b", txt))
        has_title = len(txt.split()) > 5
        has_venue = any(k in txt for k in ["journal", "conference", "doi", "ieee", "nature", "springer"])
        if all([has_author, has_year, has_title, has_venue]):
            complete_refs += 1
    completeness = complete_refs / len(references)
    if completeness < 0.7:
        score -= 0.4
    elif completeness < 0.9:
        score -= 0.1
    return min(1.0, max(0.0, score))


# ----------------------------- ACCESSIBILITY CHECK -----------------------------

def _check_reference_accessibility(references: List) -> float:
    if not references:
        return 0.1
    successes, tested = 0, 0
    for ref in references[:10]:  # limit CrossRef requests
        if not ref.doi:
            continue
        tested += 1
        try:
            r = requests.get(f"{CROSSREF_API}{ref.doi}", timeout=3)
            if r.status_code == 200:
                successes += 1
        except Exception:
            continue
    if tested == 0:
        return 0.4
    ratio = successes / tested
    return max(0.0, min(1.0, ratio))


# ----------------------------- DENSITY CHECK -----------------------------

def _check_citation_density(text: str) -> float:
    citations = re.findall(r"\([^)]*\d{4}[^)]*\)|\[[^\]]*\d{4}[^\]]*\]|\[\d+\]", text)
    words = len(re.findall(r"\b\w+\b", text))
    if not words:
        return 0.0
    density = (len(citations) / words) * 100
    if density < 1:
        return 0.3
    elif density < 2:
        return 0.6
    elif density <= 5:
        return 1.0
    elif density <= 8:
        return 0.8
    return 0.6


# ----------------------------- RECENCY CHECK -----------------------------

def _check_reference_recency(references: List) -> float:
    if not references:
        return 0.1
    years = []
    for ref in references:
        found = re.findall(r"\b(19|20)\d{2}\b", ref.text)
        years.extend([int("".join(y)) for y in found])
    if not years:
        return 0.3
    current = datetime.now().year
    recent = sum(1 for y in years if y >= current - 5)
    ratio = recent / len(years)
    if ratio < 0.3:
        return 0.4
    elif ratio < 0.5:
        return 0.7
    return 1.0


# ----------------------------- DIVERSITY CHECK -----------------------------

def _check_reference_diversity(references: List) -> float:
    if not references:
        return 0.2
    venues = [v for ref in references for v in re.findall(r"(journal|conference|arxiv|springer|nature)", ref.text.lower())]
    if not venues:
        return 0.4
    diversity = len(set(venues)) / len(venues)
    return min(1.0, diversity + 0.3)


# ----------------------------- FEEDBACK -----------------------------

def _generate_reference_feedback(*scores) -> str:
    labels = ["format", "quality", "access", "density", "recency", "diversity"]
    feedbacks = []
    for label, s in zip(labels, scores):
        if s < 0.6:
            if label == "format":
                feedbacks.append("Citations format inconsistent or incorrect style.")
            elif label == "quality":
                feedbacks.append("Incomplete or low-quality references detected.")
            elif label == "access":
                feedbacks.append("Some references are not accessible or invalid DOIs.")
            elif label == "density":
                feedbacks.append("Citation density too low relative to text length.")
            elif label == "recency":
                feedbacks.append("References are outdated; add recent research.")
            elif label == "diversity":
                feedbacks.append("Low diversity of sources; consider broader range.")
    if not feedbacks:
        return "Excellent reference structure, consistent citation format and good accessibility."
    return " ".join(feedbacks)
