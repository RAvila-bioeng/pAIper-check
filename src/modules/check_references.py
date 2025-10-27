"""
References and Citation Verification Module
Guarantees the work's strength and bibliographical support.

This module provides comprehensive analysis of:
- Automated validation of bibliographic citations format
- Existence and accessibility verification of cited sources
- Credibility and relevance analysis of references
- Citation density and distribution analysis
- Reference diversity and source quality assessment
- Semantic relevance evaluation (optional extension)

Key Features:
- DOI validation via CrossRef API
- Multiple citation style support (APA, IEEE, Vancouver, etc.)
- Academic source credibility assessment
- Reference recency and impact analysis
- Comprehensive feedback system for improvement
"""

import re
import requests
import json
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from models.score import PillarResult
from models.paper import Reference

# API Configuration
CROSSREF_API = "https://api.crossref.org/works/"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/"
USER_AGENT = "pAIper-check/1.0 (https://github.com/your-repo/pAIper-check)"

# Cache for API responses to improve performance
_api_cache = {}
_cache_expiry = 3600  # 1 hour


def evaluate(paper, use_gpt=False) -> dict:
    """
    Evaluate references and citation quality using advanced logic.
    
    This function implements the References and Citation Verification pillar,
    ensuring the work's strength and bibliographical support through:
    - Format correctness validation
    - Source accessibility verification (via DOI CrossRef)
    - Credibility and recency analysis
    - Diversity of references assessment
    - Semantic relevance evaluation
    
    Args:
        paper: Paper object containing full_text and references
        
    Returns:
        dict: PillarResult containing score and detailed feedback
    """
    text = paper.full_text
    references = paper.references

    # Core evaluation metrics
    format_score = _check_citation_format(text)
    quality_score = _check_reference_quality(references)
    accessibility_score = _check_reference_accessibility(references)
    density_score = _check_citation_density(text)
    recency_score = _check_reference_recency(references)
    diversity_score = _check_reference_diversity(references)
    credibility_score = _check_reference_credibility(references)
    semantic_score = _check_semantic_relevance(text, references)

    # Calculate weighted overall score
    weights = {
        'format': 0.15,
        'quality': 0.20,
        'accessibility': 0.15,
        'density': 0.10,
        'recency': 0.15,
        'diversity': 0.10,
        'credibility': 0.10,
        'semantic': 0.05
    }
    
    overall_score = (
        format_score * weights['format'] +
        quality_score * weights['quality'] +
        accessibility_score * weights['accessibility'] +
        density_score * weights['density'] +
        recency_score * weights['recency'] +
        diversity_score * weights['diversity'] +
        credibility_score * weights['credibility'] +
        semantic_score * weights['semantic']
    )

    feedback = _generate_reference_feedback(
        format_score, quality_score, accessibility_score, density_score, 
        recency_score, diversity_score, credibility_score, semantic_score
    )
    # 🔹 Si se usa la opción --use-chatGPT, activar el análisis avanzado con Perplexity
    if use_gpt:
        try:
            from integrations.perplexity_api import analyze_references
            gpt_feedback = analyze_references(references)
            feedback += f"\n\n🔍 Perplexity Sonar Pro Analysis:\n{gpt_feedback}"
        except ImportError:
            feedback += f"\n\n⚠️ Perplexity integration not found. Please ensure 'integrations/perplexity_api.py' exists."
        except Exception as e:
            feedback += f"\n\n⚠️ Perplexity analysis failed: {e}"
    return PillarResult("References & Citations", overall_score, feedback).__dict__


# ----------------------------- FORMAT CHECK -----------------------------

def _check_citation_format(text: str) -> float:
    score = 1.0
    
    # More comprehensive citation patterns
    citation_patterns = [
        r"\([^)]*\d{4}[^)]*\)",           # (Author, 2024)
        r"\[\d+\]",                       # [1]
        r"\[[^\]]*\d{4}[^\]]*\]",         # [Author, 2023]
        r"\([^)]*\d{4}[^)]*\)",           # (Smith et al., 2024)
        r"\[[A-Za-z\s]+,\s*\d{4}\]",      # [Smith, 2024]
        r"\([A-Za-z\s]+,\s*\d{4}\)",      # (Smith, 2024)
        r"\[[^\]]*\d{4}[^\]]*\]",         # [Smith et al., 2024]
    ]
    
    citations = []
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, text, re.IGNORECASE))
    
    if not citations:
        return 0.1
    
    # Check for different citation styles
    styles = {
        "parenthetical": len(re.findall(r"\([^)]*\d{4}[^)]*\)", text, re.IGNORECASE)),
        "numbered": len(re.findall(r"\[\d+\]", text)),
        "bracketed": len(re.findall(r"\[[^\]]*\d{4}[^\]]*\]", text, re.IGNORECASE)),
    }
    
    # Remove zero counts
    styles = {k: v for k, v in styles.items() if v > 0}
    
    if not styles:
        return 0.2
    
    dominant_style = max(styles, key=styles.get)
    total = sum(styles.values())
    consistency = styles[dominant_style] / total if total > 0 else 0
    
    if consistency < 0.75:
        score -= 0.3
    
    # Check for misplaced citations (after periods followed by capital letters)
    misplaced = len(re.findall(r"\.\s*\([^)]*\d{4}[^)]*\)\s*[A-Z]", text))
    if misplaced > len(citations) * 0.25:
        score -= 0.2

    return max(0.0, score)


# ----------------------------- QUALITY CHECK -----------------------------

def _check_reference_quality(references: List[Reference]) -> float:
    if not references:
        return 0.1
    
    score = 1.0
    complete_refs = 0
    
    for ref in references:
        if not ref.text or not ref.text.strip():
            continue
            
        txt = ref.text.lower()
        
        # More robust author detection
        has_author = bool(re.findall(r"[A-Z][a-z]+,\s*[A-Z]\.|et\s+al\.|and\s+[A-Z][a-z]+|[A-Z]\.\s*[A-Z][a-z]+|[A-Z][a-z]+\s+[A-Z]\.", ref.text, re.IGNORECASE))
        
        # Year detection (1900-2099)
        has_year = bool(re.search(r"\b(19|20)\d{2}\b", txt))
        
        # Title detection (more than 3 words suggests a title)
        has_title = len(txt.split()) > 3
        
        # Venue detection (expanded list)
        venue_keywords = [
            "journal", "conference", "doi", "ieee", "nature", "springer", 
            "proceedings", "symposium", "workshop", "arxiv", "pubmed",
            "acm", "elsevier", "wiley", "taylor", "francis", "plos"
        ]
        has_venue = any(keyword in txt for keyword in venue_keywords)
        
        components = [has_author, has_year, has_title, has_venue]
        if sum(components) >= 3:  # Al menos 3 componentes presentes
            complete_refs += 1
    
    if len(references) == 0:
        return 0.1
        
    completeness = complete_refs / len(references)
    
    if completeness < 0.5:
        score -= 0.4
    elif completeness < 0.7:
        score -= 0.2
    elif completeness < 0.9:
        score -= 0.1
    
    return min(1.0, max(0.0, score))


# ----------------------------- ACCESSIBILITY CHECK -----------------------------

def _check_reference_accessibility(references: List[Reference]) -> float:
    """
    Verify accessibility and validity of cited sources via DOI CrossRef.
    
    Args:
        references: List of Reference objects
        
    Returns:
        float: Score between 0.0 and 1.0
    """
    if not references:
        return 0.1
    
    successes, tested = 0, 0
    
    for ref in references[:15]:  # Increased limit for better sampling
        if not ref.doi or not ref.doi.strip():
            continue
            
        # Validate DOI format first
        doi = ref.doi.strip()
        if not _validate_doi_format(doi):
            continue
            
        tested += 1
        
        # Use cached API request
        api_url = f"{CROSSREF_API}{doi}"
        response_data = _make_api_request(api_url)
        
        if response_data:
            successes += 1
        # If response_data is None, it means the DOI is not accessible
    
    if tested == 0:
        return 0.4
    
    ratio = successes / tested
    return max(0.0, min(1.0, ratio))


# ----------------------------- DENSITY CHECK -----------------------------

def _check_citation_density(text: str) -> float:
    # More comprehensive citation detection
    citation_patterns = [
        r"\([^)]*\d{4}[^)]*\)",           # (Author, 2024)
        r"\[\d+\]",                       # [1]
        r"\[[^\]]*\d{4}[^\]]*\]",         # [Author, 2023]
    ]
    
    citations = []
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Count words more accurately (excluding citations)
    words = len(re.findall(r"\b\w+\b", text))
    if not words:
        return 0.0
    
    # Calculate density as citations per 100 words
    density = (len(citations) / words) * 100
    
    # More nuanced scoring based on academic standards
    if density < 0.5:
        return 0.2  # Very low citation density
    elif density < 1.0:
        return 0.4  # Low citation density
    elif density < 2.0:
        return 0.6  # Below average
    elif density <= 5.0:
        return 1.0  # Optimal range
    elif density <= 8.0:
        return 0.8  # High but acceptable
    elif density <= 12.0:
        return 0.6  # Very high
    else:
        return 0.4  # Excessive citations


# ----------------------------- RECENCY CHECK -----------------------------

def _check_reference_recency(references: List[Reference]) -> float:
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

def _check_reference_diversity(references: List[Reference]) -> float:
    if not references:
        return 0.2
    
    # Extract venue information from references
    venues = []
    publishers = []
    
    for ref in references:
        if not ref.text:
            continue
            
        txt = ref.text.lower()
        
        # Extract venue types
        venue_matches = re.findall(r"(journal|conference|proceedings|symposium|workshop|arxiv|preprint)", txt)
        venues.extend(venue_matches)
        
        # Extract publisher information
        publisher_matches = re.findall(r"(springer|nature|ieee|acm|elsevier|wiley|taylor|francis|plos|pubmed)", txt)
        publishers.extend(publisher_matches)
    
    if not venues and not publishers:
        return 0.3
    
    # Calculate diversity based on both venues and publishers
    total_sources = venues + publishers
    if not total_sources:
        return 0.3
        
    diversity = len(set(total_sources)) / len(total_sources)
    
    # Bonus for having both venue and publisher diversity
    if venues and publishers:
        diversity += 0.1
    
    return min(1.0, diversity)


# ----------------------------- CREDIBILITY CHECK -----------------------------

def _check_reference_credibility(references: List[Reference]) -> float:
    """
    Assess the credibility and authority of reference sources.
    
    Args:
        references: List of Reference objects
        
    Returns:
        float: Score between 0.0 and 1.0
    """
    if not references:
        return 0.1
    
    score = 1.0
    credible_sources = 0
    
    # High-impact journal patterns
    high_impact_journals = [
        'nature', 'science', 'cell', 'lancet', 'nejm', 'jama', 'bmj',
        'pnas', 'plos one', 'ieee', 'acm', 'springer', 'elsevier',
        'wiley', 'taylor francis', 'oxford', 'cambridge'
    ]
    
    # Prestigious conference patterns
    prestigious_conferences = [
        'neurips', 'icml', 'iclr', 'aaai', 'ijcai', 'acl', 'emnlp',
        'cvpr', 'iccv', 'eccv', 'sigcomm', 'nsdi', 'osdi', 'sosp',
        'chi', 'uist', 'ubicomp', 'www', 'kdd', 'icdm'
    ]
    
    for ref in references:
        if not ref.text:
            continue
            
        txt = ref.text.lower()
        ref_score = 0.0
        
        # Check for high-impact journals
        for journal in high_impact_journals:
            if journal in txt:
                ref_score += 0.4
                break
        
        # Check for prestigious conferences
        for conf in prestigious_conferences:
            if conf in txt:
                ref_score += 0.3
                break
        
        # Check for DOI presence (indicates formal publication)
        if ref.doi and ref.doi.strip():
            ref_score += 0.2
        
        # Check for peer-reviewed indicators
        peer_review_indicators = [
            'peer reviewed', 'refereed', 'reviewed', 'journal',
            'proceedings', 'conference', 'symposium'
        ]
        
        for indicator in peer_review_indicators:
            if indicator in txt:
                ref_score += 0.1
                break
        
        # Check for author credentials (professor, dr, phd)
        author_credentials = ['prof.', 'professor', 'dr.', 'phd', 'md']
        for credential in author_credentials:
            if credential in txt:
                ref_score += 0.1
                break
        
        # Normalize score
        ref_score = min(1.0, ref_score)
        
        if ref_score >= 0.6:
            credible_sources += 1
    
    if len(references) == 0:
        return 0.1
        
    credibility_ratio = credible_sources / len(references)
    
    if credibility_ratio < 0.5:
        score -= 0.4
    elif credibility_ratio < 0.7:
        score -= 0.2
    elif credibility_ratio < 0.8:
        score -= 0.1
    
    return max(0.0, min(1.0, score))


# ----------------------------- SEMANTIC RELEVANCE CHECK -----------------------------

def _check_semantic_relevance(text: str, references: List[Reference]) -> float:
    """
    Assess semantic relevance of references to the paper content.
    
    Args:
        text: Full text of the paper
        references: List of Reference objects
        
    Returns:
        float: Score between 0.0 and 1.0
    """
    if not references or not text:
        return 0.3
    
    # Extract key terms from the paper
    paper_keywords = _extract_paper_keywords(text)
    
    if not paper_keywords:
        return 0.3
    
    relevant_refs = 0
    
    for ref in references:
        if not ref.text:
            continue
            
        ref_keywords = _extract_reference_keywords(ref.text)
        
        # Calculate keyword overlap
        overlap = len(set(paper_keywords) & set(ref_keywords))
        relevance_score = overlap / max(len(paper_keywords), 1)
        
        if relevance_score >= 0.3:  # At least 30% keyword overlap
            relevant_refs += 1
    
    if len(references) == 0:
        return 0.3
        
    relevance_ratio = relevant_refs / len(references)
    
    if relevance_ratio < 0.4:
        return 0.3
    elif relevance_ratio < 0.6:
        return 0.6
    elif relevance_ratio < 0.8:
        return 0.8
    else:
        return 1.0


def _extract_paper_keywords(text: str) -> List[str]:
    """Extract key terms from paper text."""
    # Remove common words and extract meaningful terms
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
        'her', 'its', 'our', 'their', 'paper', 'study', 'research', 'work',
        'method', 'approach', 'result', 'conclusion', 'introduction'
    }
    
    # Extract words (3+ characters, alphanumeric)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out common words and return unique terms
    keywords = list(set(word for word in words if word not in common_words))
    
    # Return top 20 most frequent keywords
    word_counts = {}
    for word in words:
        if word not in common_words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:20]]


def _extract_reference_keywords(ref_text: str) -> List[str]:
    """Extract key terms from reference text."""
    # Similar to paper keywords but focused on reference content
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'journal', 'conference', 'proceedings',
        'symposium', 'workshop', 'paper', 'study', 'research', 'work',
        'method', 'approach', 'result', 'conclusion', 'introduction'
    }
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', ref_text.lower())
    keywords = list(set(word for word in words if word not in common_words))
    
    return keywords[:10]  # Top 10 keywords from reference


# ----------------------------- UTILITY FUNCTIONS -----------------------------

def _make_api_request(url: str, timeout: int = 5) -> Optional[Dict]:
    """
    Make API request with caching and error handling.
    
    Args:
        url: API endpoint URL
        timeout: Request timeout in seconds
        
    Returns:
        Dict or None: API response data or None if failed
    """
    # Check cache first
    cache_key = url
    if cache_key in _api_cache:
        cached_data, timestamp = _api_cache[cache_key]
        if time.time() - timestamp < _cache_expiry:
            return cached_data
    
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            # Cache successful response
            _api_cache[cache_key] = (data, time.time())
            return data
        else:
            return None
            
    except Exception:
        return None


def _validate_doi_format(doi: str) -> bool:
    """
    Validate DOI format according to ISO 26324.
    
    Args:
        doi: DOI string to validate
        
    Returns:
        bool: True if valid DOI format
    """
    if not doi or not isinstance(doi, str):
        return False
    
    doi = doi.strip()
    
    # Basic DOI pattern: 10.xxxx/xxxx
    doi_pattern = r'^10\.\d{4,}/.+'
    
    return bool(re.match(doi_pattern, doi))


# ----------------------------- FEEDBACK -----------------------------

def _generate_reference_feedback(format_score, quality_score, accessibility_score, density_score, 
                                 recency_score, diversity_score, credibility_score, semantic_score) -> str:
    """Generate detailed and balanced feedback for reference evaluation."""
    feedback_parts = []

    # 1. Format
    if format_score >= 0.8:
        feedback_parts.append("✓ Citation format is consistent and follows a recognizable academic style.")
    else:
        feedback_parts.append("⚠️ Citation format may be inconsistent. Ensure a single, standard style (e.g., APA, IEEE) is used throughout.")

    # 2. Quality/Completeness
    if quality_score >= 0.8:
        feedback_parts.append("✓ References are well-formed and contain sufficient bibliographic details (author, year, title, etc.).")
    else:
        feedback_parts.append("⚠️ Some references appear incomplete. Ensure all entries include author, year, title, and publication venue.")

    # 3. Accessibility (DOI check)
    if accessibility_score >= 0.8:
        feedback_parts.append("✓ Most cited DOIs were successfully verified and are accessible.")
    elif accessibility_score >= 0.5:
        feedback_parts.append("✓ A good portion of cited DOIs were verified. Some could not be found.")
    else:
        feedback_parts.append("⚠️ Many cited DOIs could not be verified via CrossRef. Check for typos or consider citing more stable sources.")

    # 4. Density
    if 0.6 < density_score < 1.0: # 0.8-1.0 range
        feedback_parts.append("✓ Citation density is appropriate for a scientific paper.")
    else:
        feedback_parts.append("⚠️ Citation density seems low. Ensure all claims are adequately supported by references.")

    # 5. Recency
    if recency_score >= 0.8:
        feedback_parts.append("✓ The bibliography includes a good proportion of recent sources, showing engagement with current research.")
    else:
        feedback_parts.append("⚠️ The bibliography may be outdated. It's advisable to include more research from the last 5 years.")

    # 6. Diversity
    if diversity_score >= 0.7:
        feedback_parts.append("✓ Good diversity of sources from various journals and publishers.")
    else:
        feedback_parts.append("⚠️ Source diversity could be improved. Consider citing a broader range of venues and publication types.")
        
    # 7. Credibility
    if credibility_score >= 0.8:
        feedback_parts.append("✓ References are from credible, high-impact sources.")
    else:
        feedback_parts.append("⚠️ The credibility of some sources could be stronger. Prioritize peer-reviewed journals and reputable conferences.")

    # 8. Semantic Relevance
    if semantic_score >= 0.7:
        feedback_parts.append("✓ Cited works appear to be semantically relevant to the paper's main topics.")
    else:
        feedback_parts.append("⚠️ Some references may lack direct relevance. Ensure every citation strongly supports the specific point being made.")

    return "\n  ".join(feedback_parts)


# Esto sirve para probar el análisis de referencias con Perplexity desde la terminal
if __name__ == "__main__":
    import argparse
    from src.perplexity_integration import analyze_references

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-chatGPT", action="store_true", help="Enable Perplexity reference analysis")
    args = parser.parse_args()

    if args.use_chatGPT:
        from models.paper import Paper
        import json

        sample_paper = Paper(full_text="Sample text", references=[])

        print("🔍 Running Perplexity Sonar Pro analysis on references...")
        try:
            analysis = analyze_references(sample_paper.references)
            print("\n🧠 Perplexity Sonar Pro feedback:\n", analysis)
        except Exception as e:
            print(f"❌ Error using Perplexity API: {e}")