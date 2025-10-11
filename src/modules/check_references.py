"""
References and Citation Verification Module
Guarantees the work's strength and bibliographical support.
"""

import re
from typing import List, Tuple
from models.score import PillarResult


def evaluate(paper) -> dict:
    """
    Evaluate references and citation quality.
    
    Args:
        paper: Paper object with references and text content
        
    Returns:
        dict: Score and feedback for references quality
    """
    text = paper.full_text
    references = paper.references
    
    # Check citation format
    format_score = _check_citation_format(text)
    
    # Check reference quality
    quality_score = _check_reference_quality(references)
    
    # Check citation density
    density_score = _check_citation_density(text)
    
    # Check reference recency
    recency_score = _check_reference_recency(references)
    
    # Check reference diversity
    diversity_score = _check_reference_diversity(references)
    
    # Calculate overall score
    overall_score = (format_score + quality_score + density_score + recency_score + diversity_score) / 5
    
    # Generate feedback
    feedback = _generate_reference_feedback(format_score, quality_score, density_score, recency_score, diversity_score)
    
    return PillarResult("References & Citations", overall_score, feedback).__dict__


def _check_citation_format(text: str) -> float:
    """Checks the correct notation and format of bibliographic citations."""
    score = 1.0
    
    # Common citation patterns
    citation_patterns = [
        r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023) or (Smith et al., 2023)
        r'\[[^\]]*\d{4}[^\]]*\]',  # [1] or [Author, 2023]
        r'\d+\s*\[[^\]]*\]',  # 1 [Author, 2023]
        r'\[\d+\]',  # [1], [2], etc.
    ]
    
    citations = []
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, text))
    
    if not citations:
        return 0.2  # No citations found
    
    # Check for consistent citation style
    styles = {
        'parenthetical': len(re.findall(r'\([^)]*\d{4}[^)]*\)', text)),
        'numbered': len(re.findall(r'\[\d+\]', text)),
        'bracketed': len(re.findall(r'\[[^\]]*\d{4}[^\]]*\]', text)),
    }
    
    # Find the dominant style
    dominant_style = max(styles, key=styles.get)
    total_citations = sum(styles.values())
    
    if total_citations > 0:
        consistency_ratio = styles[dominant_style] / total_citations
        if consistency_ratio < 0.8:
            score -= 0.3  # Inconsistent citation style
    
    # Check for proper citation placement
    # Citations should not be at the beginning of sentences
    sentence_start_citations = len(re.findall(r'\.\s*\([^)]*\d{4}[^)]*\)\s*[A-Z]', text))
    if sentence_start_citations > len(citations) * 0.3:
        score -= 0.2
    
    return max(0.0, score)


def _check_reference_quality(references: List) -> float:
    """Check the quality and credibility of references."""
    if not references:
        return 0.1
    
    score = 1.0
    
    # Check for complete reference information
    complete_refs = 0
    for ref in references:
        ref_text = ref.text.lower()
        
        # Check for essential elements
        has_author = any(word in ref_text for word in ['author', 'et al', 'smith', 'jones', 'brown', 'wilson', 'taylor'])
        has_year = bool(re.search(r'\d{4}', ref_text))
        has_title = len(ref_text.split()) > 5  # Reasonable title length
        has_venue = any(word in ref_text for word in ['journal', 'conference', 'proceedings', 'arxiv', 'doi', 'http'])
        
        if has_author and has_year and has_title and has_venue:
            complete_refs += 1
    
    completeness_ratio = complete_refs / len(references)
    if completeness_ratio < 0.7:
        score -= 0.4
    elif completeness_ratio < 0.9:
        score -= 0.2
    
    # Check for DOI presence (indicates quality)
    doi_count = sum(1 for ref in references if ref.doi)
    doi_ratio = doi_count / len(references)
    
    if doi_ratio < 0.3:
        score -= 0.2
    elif doi_ratio > 0.7:
        score += 0.1
    
    # Check for peer-reviewed sources
    peer_reviewed_indicators = [
        'journal', 'ieee', 'acm', 'springer', 'elsevier', 'nature', 'science',
        'proceedings', 'conference', 'symposium', 'workshop'
    ]
    
    peer_reviewed_count = 0
    for ref in references:
        ref_text = ref.text.lower()
        if any(indicator in ref_text for indicator in peer_reviewed_indicators):
            peer_reviewed_count += 1
    
    peer_reviewed_ratio = peer_reviewed_count / len(references)
    if peer_reviewed_ratio < 0.5:
        score -= 0.3
    elif peer_reviewed_ratio > 0.8:
        score += 0.1
    
    return min(1.0, max(0.0, score))


def _check_citation_density(text: str) -> float:
    """Check if there are sufficient citations for the content."""
    # Count citations
    citations = re.findall(r'\([^)]*\d{4}[^)]*\)|\[[^\]]*\d{4}[^\]]*\]|\[\d+\]', text)
    citation_count = len(citations)
    
    # Count words
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    
    if word_count == 0:
        return 0.0
    
    # Calculate citation density (citations per 100 words)
    citation_density = (citation_count / word_count) * 100
    
    # Academic papers typically have 2-5 citations per 100 words
    if citation_density < 1.0:
        return 0.3  # Too few citations
    elif citation_density < 2.0:
        return 0.6  # Below average
    elif citation_density <= 5.0:
        return 1.0  # Good range
    elif citation_density <= 8.0:
        return 0.8  # Slightly high but acceptable
    else:
        return 0.6  # Too many citations


def _check_reference_recency(references: List) -> float:
    """Check the currency and relevance of references."""
    if not references:
        return 0.1
    
    import datetime
    current_year = datetime.datetime.now().year
    
    # Extract years from references
    years = []
    for ref in references:
        year_matches = re.findall(r'\b(19|20)\d{2}\b', ref.text)
        if year_matches:
            years.extend([int(year) for year in year_matches])
    
    if not years:
        return 0.3  # No years found
    
    # Calculate recency metrics
    recent_years = [year for year in years if year >= current_year - 5]  # Last 5 years
    very_recent = [year for year in years if year >= current_year - 2]   # Last 2 years
    
    recent_ratio = len(recent_years) / len(years)
    very_recent_ratio = len(very_recent) / len(years)
    
    score = 1.0
    
    # Penalize if too few recent references
    if recent_ratio < 0.3:
        score -= 0.4
    elif recent_ratio < 0.5:
        score -= 0.2
    
    # Bonus for very recent references
    if very_recent_ratio > 0.2:
        score += 0.1
    
    # Check for very old references (might indicate outdated work)
    old_years = [year for year in years if year < current_year - 10]
    old_ratio = len(old_years) / len(years)
    
    if old_ratio > 0.5:
        score -= 0.2
    
    return min(1.0, max(0.0, score))


def _check_reference_diversity(references: List) -> float:
    """Check the diversity and breadth of references."""
    if not references:
        return 0.1
    
    if len(references) < 5:
        return 0.5  # Too few references to assess diversity
    
    score = 1.0
    
    # Check for author diversity (avoid self-citation dominance)
    authors = []
    for ref in references:
        # Extract potential author names (simplified)
        ref_text = ref.text
        # Look for patterns like "Smith, J." or "Smith and Jones"
        author_matches = re.findall(r'\b[A-Z][a-z]+,\s*[A-Z]\.?\b', ref_text)
        authors.extend(author_matches)
    
    if authors:
        unique_authors = len(set(authors))
        author_diversity = unique_authors / len(authors)
        
        if author_diversity < 0.7:
            score -= 0.3
        elif author_diversity > 0.9:
            score += 0.1
    
    # Check for venue diversity
    venues = []
    venue_keywords = ['journal', 'conference', 'proceedings', 'symposium', 'workshop', 'arxiv']
    
    for ref in references:
        ref_text = ref.text.lower()
        for keyword in venue_keywords:
            if keyword in ref_text:
                venues.append(keyword)
                break
    
    if venues:
        unique_venues = len(set(venues))
        venue_diversity = unique_venues / len(venues)
        
        if venue_diversity < 0.5:
            score -= 0.2
    
    # Check for geographical diversity (simplified)
    countries = []
    country_keywords = ['usa', 'united states', 'china', 'germany', 'uk', 'united kingdom', 'france', 'japan', 'canada', 'australia']
    
    for ref in references:
        ref_text = ref.text.lower()
        for keyword in country_keywords:
            if keyword in ref_text:
                countries.append(keyword)
                break
    
    if countries:
        unique_countries = len(set(countries))
        if unique_countries > 3:
            score += 0.1
    
    return min(1.0, max(0.0, score))


def _generate_reference_feedback(format_score: float, quality_score: float, 
                               density_score: float, recency_score: float, 
                               diversity_score: float) -> str:
    """Generate detailed feedback for references and citations."""
    feedback_parts = []
    
    if format_score < 0.7:
        feedback_parts.append("Citation format needs improvement. Ensure consistent citation style throughout.")
    
    if quality_score < 0.7:
        feedback_parts.append("Reference quality could be improved. Include more peer-reviewed sources and complete bibliographic information.")
    
    if density_score < 0.6:
        feedback_parts.append("Citation density is low. Add more references to support your claims.")
    elif density_score > 0.8:
        feedback_parts.append("Citation density is high. Consider if all citations are necessary.")
    
    if recency_score < 0.6:
        feedback_parts.append("Include more recent references to show awareness of current research.")
    
    if diversity_score < 0.7:
        feedback_parts.append("Reference diversity could be improved. Include sources from different authors and venues.")
    
    if not feedback_parts:
        feedback_parts.append("Good reference quality with appropriate citation density, recency, and diversity.")
    
    return " ".join(feedback_parts)
