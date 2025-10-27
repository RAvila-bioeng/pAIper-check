"""
Coherence and Cohesion Analysis Module - IMPROVED VERSION
Evaluates the flow and logical connection of the argument.
"""

import re
import json
from typing import List, Dict, Tuple
from collections import Counter
from models.score import PillarResult
from modules.gpt_cohesion_analyzer import enhance_coherence_with_gpt

# --- Load Keywords from External Configuration ---
try:
    with open('config/keywords.json', 'r') as f:
        _keywords = json.load(f)
    STOP_WORDS = set(_keywords.get('STOP_WORDS', [])) # Use a set for efficient lookup
    CONNECTORS = _keywords.get('CONNECTORS', {})
    TRANSITION_PHRASES = _keywords.get('TRANSITION_PHRASES', [])
    SECTION_KEYWORDS = _keywords.get('SECTION_KEYWORDS', {})
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Warning: Could not load keywords from config/keywords.json. Using empty fallbacks. Error: {e}")
    STOP_WORDS, CONNECTORS, TRANSITION_PHRASES, SECTION_KEYWORDS = set(), {}, [], {}


def evaluate(paper, use_gpt: bool = False) -> dict:
    """
    Evaluate coherence and cohesion of the paper.
    
    Args:
        paper: Paper object with text content
        
    Returns:
        dict: Score and feedback for coherence and cohesion
    """
    text = paper.full_text
    sections = paper.sections if hasattr(paper, 'sections') else []
    
    # Calculate text metrics for normalization
    text_metrics = _calculate_text_metrics(text)
    
    # Check argumentative fluency
    fluency_score, fluency_details = _check_argumentative_fluency(text, text_metrics)
    
    # Check connectivity between sections
    connectivity_score, connectivity_details = _check_section_connectivity(text, sections, text_metrics)
    
    # Check narrative consistency
    consistency_score, consistency_details = _check_narrative_consistency(text, text_metrics)
    
    # Check logical flow
    flow_score, flow_details = _check_logical_flow_refactored(sections)
    
    # Calculate weighted overall score
    weights = {
        'fluency': 0.3,
        'connectivity': 0.25,
        'consistency': 0.25,
        'flow': 0.2
    }
    
    overall_score = (
        fluency_score * weights['fluency'] +
        connectivity_score * weights['connectivity'] +
        consistency_score * weights['consistency'] +
        flow_score * weights['flow']
    )
    
    # Generate detailed feedback
    feedback = _generate_cohesion_feedback(
        fluency_score, fluency_details,
        connectivity_score, connectivity_details,
        consistency_score, consistency_details,
        flow_score, flow_details
    )
    
    basic_result = PillarResult("Coherence & Cohesion", overall_score, feedback).__dict__

    # Add detailed score breakdown for transparency
    basic_result['score_breakdown'] = {
        'Argumentative Fluency': fluency_score,
        'Section Connectivity': connectivity_score,
        'Narrative Consistency': consistency_score,
        'Logical Flow': flow_score
    }
    
    if use_gpt:
        gpt_analysis_data = {
            'paper_info': {
                'title': paper.title,
                'word_count': text_metrics['word_count']
            },
            'structural_metrics': {
                'connector_density': fluency_details.get('connector_density', 0),
                'transition_ratio': fluency_details.get('transition_ratio', 0),
                'terminology_inconsistencies': consistency_details.get('terminology_inconsistencies', 0),
                'order_violations': flow_details.get('order_violations', 0)
            },
            'problematic_areas': [], 
            'key_sections': {s.title: s.content for s in sections[:2]} if sections else {}
        }
        basic_result['gpt_analysis_data'] = gpt_analysis_data
        return enhance_coherence_with_gpt(paper, basic_result, force_analysis=True)
    
    return basic_result


def _calculate_text_metrics(text: str) -> Dict:
    """Calculate basic text metrics for normalization."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'paragraph_count': len(paragraphs),
        'avg_sentence_length': len(words) / max(1, len(sentences)),
        'avg_paragraph_length': len(sentences) / max(1, len(paragraphs))
    }


def _check_argumentative_fluency(text: str, metrics: Dict) -> Tuple[float, Dict]:
    """Check how well ideas are developed throughout the manuscript."""
    score = 1.0
    details = {}
    
    # Count connectors by category
    connector_counts = {}
    total_connectors = 0
    
    for category, connectors in CONNECTORS.items():
        count = sum(len(re.findall(rf'\b{re.escape(connector)}\b', text.lower())) 
                   for connector in connectors)
        connector_counts[category] = count
        total_connectors += count
    
    details['connector_counts'] = connector_counts
    details['total_connectors'] = total_connectors
    
    # Normalize by text length (connectors per 1000 words)
    words_per_thousand = metrics['word_count'] / 1000
    normalized_connector_density = total_connectors / max(0.1, words_per_thousand)
    
    details['connector_density'] = normalized_connector_density
    
    # --- New Rubric for Connector Density ---
    density_score = 1.0
    if normalized_connector_density >= 8:
        density_score = 1.0
        details['fluency_issue'] = 'Excellent argumentative fluency.'
    elif 4 <= normalized_connector_density < 8:
        density_score = 0.8
        details['fluency_issue'] = 'Good argumentative fluency.'
    elif 2 <= normalized_connector_density < 4:
        density_score = 0.6
        details['fluency_issue'] = 'Argumentative fluency could be improved. The text is a bit dry.'
    else: # < 2
        density_score = 0.3
        details['fluency_issue'] = 'Poor argumentative fluency, potential logical gaps.'

    score = density_score
    
    # Other factors penalize the base score from the density rubric
    # Check diversity of connectors
    categories_used = sum(1 for count in connector_counts.values() if count > 0)
    if categories_used < 3:
        score -= 0.15 # Reduced penalty
        details['diversity_issue'] = f'Only {categories_used}/5 connector types used'
    
    # Check paragraph transitions
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    transitions_found = 0
    
    for i in range(1, len(paragraphs)):
        para = paragraphs[i]
        first_sentence = para.split('.')[0].lower()
        if any(connector in first_sentence for category in CONNECTORS.values() for connector in category):
            transitions_found += 1
    
    transition_ratio = transitions_found / max(1, len(paragraphs) - 1)
    details['transition_ratio'] = transition_ratio
    
    if transition_ratio < 0.3:
        score -= 0.2 # Reduced penalty
        details['transition_issue'] = 'Poor paragraph transitions'
    elif transition_ratio < 0.5:
        score -= 0.1
    
    return max(0.0, min(1.0, score)), details


def _check_section_connectivity(text: str, sections: List, metrics: Dict) -> Tuple[float, Dict]:
    """Check logical relationship between paragraphs and sections."""
    score = 1.0
    details = {}
    
    if not sections or len(sections) < 2:
        details['warning'] = 'Insufficient sections for connectivity analysis'
        return 0.6, details
    
    # Extract section titles
    section_titles = [section.title.lower() for section in sections if hasattr(section, 'title')]
    details['sections_found'] = len(section_titles)
    
    # Count transition phrases
    transition_count = sum(
        len(re.findall(rf'\b{re.escape(phrase)}\b', text.lower())) 
        for phrase in TRANSITION_PHRASES
    )
    
    # Normalize by number of sections
    normalized_transitions = transition_count / max(1, len(sections) - 1)
    details['transitions_per_section'] = normalized_transitions
    
    if normalized_transitions < 1:
        score -= 0.4
        details['connectivity_issue'] = 'Very few cross-references between sections'
    elif normalized_transitions < 2:
        score -= 0.2
        details['connectivity_issue'] = 'Limited cross-references'
    
    # Check for forward/backward references
    reference_patterns = [
        r'\bas (discussed|mentioned|shown|demonstrated|described|stated) (in|above|below|previously|earlier)',
        r'\b(see|refer to|according to) (section|chapter|table|figure)',
        r'\b(following|previous|next) (section|chapter)'
    ]
    
    reference_count = sum(
        len(re.findall(pattern, text.lower(), re.IGNORECASE)) 
        for pattern in reference_patterns
    )
    
    details['reference_count'] = reference_count
    
    if reference_count == 0:
        score -= 0.2
        details['reference_issue'] = 'No explicit section references found'
    
    return max(0.0, min(1.0, score)), details


def _check_narrative_consistency(text: str, metrics: Dict) -> Tuple[float, Dict]:
    """Ensure uniform storyline from introduction to conclusions."""
    score = 1.0
    details = {}
    
    # Extract technical terms (4+ letters, not stop words)
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    term_frequency = Counter(w for w in words if w not in STOP_WORDS)
    
    # Get frequent terms (top 20 or those appearing 5+ times)
    frequent_terms = {term: count for term, count in term_frequency.items() 
                     if count >= 5}
    details['key_terms_count'] = len(frequent_terms)
    
    # Check for terminology inconsistencies
    inconsistencies = _detect_terminology_variations(frequent_terms)
    details['terminology_inconsistencies'] = len(inconsistencies)
    
    if len(inconsistencies) > len(frequent_terms) * 0.2:
        score -= 0.3
        details['consistency_issue'] = f'High terminology variation: {len(inconsistencies)} inconsistencies'
    elif len(inconsistencies) > 0:
        score -= 0.1
    
    # Check point of view consistency
    first_person = len(re.findall(r'\b(we|our|us)\b', text.lower()))
    third_person = len(re.findall(r'\b(this (study|research|paper|work|analysis))\b', text.lower()))
    passive_voice = len(re.findall(r'\b(was|were|is|are) \w+ed\b', text.lower()))
    
    total_references = first_person + third_person
    details['point_of_view'] = {
        'first_person': first_person,
        'third_person': third_person,
        'passive_voice': passive_voice
    }
    
    # Check for mixed POV (inconsistent)
    if total_references > 10:
        first_person_ratio = first_person / total_references
        if 0.2 < first_person_ratio < 0.8:
            score -= 0.2
            details['pov_issue'] = 'Inconsistent point of view (mixed first/third person)'
    
    # Check tense consistency in different sections
    past_tense = len(re.findall(r'\b\w+ed\b', text.lower()))
    present_tense = len(re.findall(r'\b(is|are|shows|demonstrates|indicates)\b', text.lower()))
    
    details['tense_usage'] = {
        'past': past_tense,
        'present': present_tense
    }
    
    return max(0.0, min(1.0, score)), details


def _detect_terminology_variations(frequent_terms: Dict) -> List[Tuple[str, str]]:
    """Detect potential terminology inconsistencies."""
    inconsistencies = []
    terms_list = list(frequent_terms.keys())
    
    for i, term1 in enumerate(terms_list):
        for term2 in terms_list[i+1:]:
            # Check for similar terms (potential inconsistencies)
            if _are_similar_terms(term1, term2):
                inconsistencies.append((term1, term2))
    
    return inconsistencies


def _are_similar_terms(term1: str, term2: str) -> bool:
    """Check if two terms are variations of each other."""
    # Check plural/singular
    if term1 + 's' == term2 or term2 + 's' == term1:
        return True
    if term1 + 'es' == term2 or term2 + 'es' == term1:
        return True
    
    # Check -tion/-ment variations
    base1 = term1.rstrip('ion').rstrip('ment').rstrip('ness')
    base2 = term2.rstrip('ion').rstrip('ment').rstrip('ness')
    if base1 == base2 and len(base1) > 3:
        return True
    
    # Check -ize/-ization variations
    if term1.replace('ization', '') == term2.replace('ize', ''):
        return True
    
    return False


def _check_logical_flow_refactored(sections: List) -> Tuple[float, Dict]:
    """
    Check the logical progression of ideas based on pre-parsed sections.
    """
    score = 1.0
    details = {}
    
    if not sections:
        return 0.4, {'warning': 'No sections found for logical flow analysis.'}

    # Map section titles to canonical names
    found_canonical_sections = []
    for section in sections:
        title_lower = section.title.lower().strip()
        matched = False
        for canonical, keywords in SECTION_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + keyword + r'\b', title_lower):
                    if canonical not in found_canonical_sections:
                        found_canonical_sections.append(canonical)
                    matched = True
                    break
            if matched:
                break
    
    details['sections_detected'] = found_canonical_sections
    
    # Expected order
    expected_order = ['introduction', 'methodology', 'results', 'discussion', 'conclusion']
    
    # Check if the order of found sections matches the expected order
    indices = [expected_order.index(s) for s in found_canonical_sections if s in expected_order]
    
    order_violations = 0
    for i in range(len(indices) - 1):
        if indices[i] > indices[i+1]:
            order_violations += 1
            
    details['order_violations'] = order_violations
    
    if order_violations > 0:
        score -= min(0.4, order_violations * 0.2)
        details['flow_issue'] = f'{order_violations} section ordering violation(s)'

    # Check for conclusion markers and link to introduction
    conclusion_section = next((s for s in sections if 'conclusion' in s.title.lower()), None)
    introduction_section = next((s for s in sections if 'introduction' in s.title.lower()), None)

    if conclusion_section:
        conclusion_text = conclusion_section.content.lower()
        conclusion_markers = ['in conclusion', 'to conclude', 'in summary', 'to summarize', 'overall', 'finally']
        conclusion_count = sum(conclusion_text.count(marker) for marker in conclusion_markers)
        details['conclusion_markers'] = conclusion_count
        if conclusion_count == 0:
            score -= 0.2
            details['conclusion_issue'] = 'No clear conclusion markers found in the conclusion section.'

        if introduction_section:
            intro_callback_phrases = ['as introduced', 'as stated', 'our objective', 'our goal', 'we aimed', 'the purpose']
            callback_found = any(phrase in conclusion_text for phrase in intro_callback_phrases)
            details['intro_conclusion_link'] = callback_found
            if not callback_found:
                score -= 0.1
    
    return max(0.0, min(1.0, score)), details


def _generate_cohesion_feedback(fluency_score: float, fluency_details: Dict,
                               connectivity_score: float, connectivity_details: Dict,
                               consistency_score: float, consistency_details: Dict,
                               flow_score: float, flow_details: Dict) -> str:
    """Generate detailed, balanced feedback for coherence and cohesion."""
    feedback_parts = []
    
    # --- Evaluate Each Sub-Pillar ---

    # 1. Argumentative Fluency
    if fluency_score >= 0.8:
        feedback_parts.append(f"✓ Excellent argumentative fluency with a good density of logical connectors ({fluency_details.get('connector_density', 0):.1f} per 1000 words).")
    elif fluency_score >= 0.6:
        feedback_parts.append(f"✓ Good argumentative fluency. Connector density is adequate ({fluency_details.get('connector_density', 0):.1f} per 1000 words).")
    else:
        feedback_parts.append(f"⚠️ Argumentative fluency is an area for improvement. The density of logical connectors is low ({fluency_details.get('connector_density', 0):.1f} per 1000 words), which can make the text feel disconnected.")
    if 'diversity_issue' in fluency_details:
        feedback_parts.append(f"   - Suggestion: Consider using a wider variety of connectors to enhance readability. {fluency_details['diversity_issue']}.")

    # 2. Section Connectivity
    if connectivity_score >= 0.8:
        feedback_parts.append("✓ Strong connectivity between sections, with clear cross-references.")
    elif connectivity_score >= 0.6:
        feedback_parts.append("✓ Adequate connectivity between sections.")
    else:
        feedback_parts.append("⚠️ Section connectivity could be stronger. The paper would benefit from more explicit links between sections (e.g., 'as discussed in Section 2').")

    # 3. Narrative Consistency
    if consistency_score >= 0.8:
        feedback_parts.append("✓ Excellent narrative consistency with consistent terminology and point of view.")
    elif consistency_score >= 0.6:
        feedback_parts.append("✓ Good narrative consistency.")
    else:
        if 'consistency_issue' in consistency_details:
            feedback_parts.append(f"⚠️ Narrative consistency needs attention. {consistency_details['consistency_issue']}.")
        if 'pov_issue' in consistency_details:
            feedback_parts.append(f"   - Suggestion: {consistency_details['pov_issue']}. Aim for a consistent voice (e.g., stick to 'we' or 'this study').")

    # 4. Logical Flow
    if flow_score >= 0.8:
        feedback_parts.append("✓ The paper follows a clear and logical structure (IMRaD).")
    elif flow_score >= 0.6:
        feedback_parts.append("✓ The overall structure follows a logical sequence.")
    else:
        if 'flow_issue' in flow_details:
            feedback_parts.append(f"⚠️ The logical flow of sections may be out of order. {flow_details['flow_issue']}. Ensure it follows the standard IMRaD structure.")
        if 'conclusion_issue' in flow_details:
            feedback_parts.append(f"   - Suggestion: {flow_details['conclusion_issue']}.")

    return "\n  ".join(feedback_parts)