"""
Coherence and Cohesion Analysis Module - IMPROVED VERSION
Evaluates the flow and logical connection of the argument.
"""

import re
from typing import List, Dict, Tuple
from collections import Counter
from models.score import PillarResult

# ============== CONFIGURACIÓN ==============
STOP_WORDS = {
    'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'said',
    'each', 'which', 'their', 'time', 'will', 'about', 'could', 'there',
    'after', 'first', 'well', 'also', 'where', 'much', 'some', 'very',
    'when', 'make', 'more', 'over', 'think', 'help', 'good', 'same',
    'many', 'most', 'other', 'new', 'way', 'may', 'say', 'use', 'man',
    'find', 'give', 'year', 'work', 'part', 'take', 'get', 'place',
    'made', 'live', 'through', 'back', 'before', 'line', 'right', 'too',
    'means', 'old', 'any', 'tell', 'boy', 'follow', 'came', 'want',
    'show', 'around', 'form', 'three', 'small', 'set', 'put', 'end',
    'why', 'again', 'turn', 'here', 'just', 'like', 'long', 'than',
    'them', 'such', 'what', 'into', 'only', 'both', 'such', 'must'
}

CONNECTORS = {
    'causality': ['therefore', 'thus', 'hence', 'consequently', 'as a result', 'for this reason'],
    'contrast': ['however', 'nevertheless', 'nonetheless', 'on the other hand', 'in contrast', 'conversely', 'whereas'],
    'addition': ['furthermore', 'moreover', 'additionally', 'in addition', 'besides', 'also'],
    'sequence': ['firstly', 'secondly', 'thirdly', 'finally', 'subsequently', 'then', 'next'],
    'conclusion': ['in conclusion', 'to conclude', 'in summary', 'overall', 'to sum up']
}

TRANSITION_PHRASES = [
    'as discussed', 'as mentioned', 'previously', 'in the previous section',
    'building on', 'extending', 'following', 'based on', 'as shown',
    'as demonstrated', 'in line with', 'consistent with', 'in accordance with'
]

SECTION_KEYWORDS = {
    'introduction': ['introduction', 'background'],
    'methodology': ['method', 'methodology', 'materials', 'experimental'],
    'results': ['result', 'finding', 'observation'],
    'discussion': ['discussion', 'interpretation'],
    'conclusion': ['conclusion', 'summary', 'final']
}


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
    flow_score, flow_details = _check_logical_flow(text, sections)
    
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
    
    return PillarResult("Coherence & Cohesion", overall_score, feedback).__dict__


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
    
    # Expected density: 10-30 connectors per 1000 words
    if normalized_connector_density < 10:
        score -= 0.4
        details['fluency_issue'] = 'Too few logical connectors'
    elif normalized_connector_density < 15:
        score -= 0.2
        details['fluency_issue'] = 'Limited logical connectors'
    elif normalized_connector_density > 40:
        score -= 0.1
        details['fluency_issue'] = 'Excessive connectors (may be verbose)'
    
    # Check diversity of connectors
    categories_used = sum(1 for count in connector_counts.values() if count > 0)
    if categories_used < 3:
        score -= 0.2
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
        score -= 0.3
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


def _check_logical_flow(text: str, sections: List) -> Tuple[float, Dict]:
    """Check the logical progression of ideas."""
    score = 1.0
    details = {}
    
    # Detect section order
    section_positions = {}
    
    for section_type, keywords in SECTION_KEYWORDS.items():
        for keyword in keywords:
            match = re.search(rf'\b{keyword}\b', text.lower())
            if match:
                section_positions[section_type] = match.start()
                break
    
    details['sections_detected'] = list(section_positions.keys())
    
    # Expected order
    expected_order = ['introduction', 'methodology', 'results', 'discussion', 'conclusion']
    found_sections = sorted(section_positions.items(), key=lambda x: x[1])
    found_order = [section for section, _ in found_sections]
    
    # Check if order matches expected
    order_violations = 0
    for i in range(len(found_order) - 1):
        current_section = found_order[i]
        next_section = found_order[i + 1]
        
        if current_section in expected_order and next_section in expected_order:
            current_idx = expected_order.index(current_section)
            next_idx = expected_order.index(next_section)
            
            if next_idx < current_idx:
                order_violations += 1
    
    details['order_violations'] = order_violations
    
    if order_violations > 0:
        score -= min(0.4, order_violations * 0.2)
        details['flow_issue'] = f'{order_violations} section ordering violation(s)'
    
    # Check for conclusion markers
    conclusion_markers = ['in conclusion', 'to conclude', 'in summary', 'to summarize', 
                         'overall', 'in summary', 'finally']
    conclusion_count = sum(text.lower().count(marker) for marker in conclusion_markers)
    
    details['conclusion_markers'] = conclusion_count
    
    if conclusion_count == 0:
        score -= 0.2
        details['conclusion_issue'] = 'No clear conclusion markers found'
    
    # Check if conclusion references introduction
    if 'conclusion' in section_positions and 'introduction' in section_positions:
        conclusion_start = section_positions['conclusion']
        conclusion_text = text[conclusion_start:].lower()
        
        intro_callback_phrases = ['as introduced', 'as stated', 'our objective', 
                                 'our goal', 'we aimed', 'the purpose']
        callback_found = any(phrase in conclusion_text for phrase in intro_callback_phrases)
        
        details['intro_conclusion_link'] = callback_found
        
        if not callback_found:
            score -= 0.1
    
    return max(0.0, min(1.0, score)), details


def _generate_cohesion_feedback(fluency_score: float, fluency_details: Dict,
                               connectivity_score: float, connectivity_details: Dict,
                               consistency_score: float, consistency_details: Dict,
                               flow_score: float, flow_details: Dict) -> str:
    """Generate detailed feedback for coherence and cohesion."""
    feedback_parts = []
    
    # Fluency feedback
    if fluency_score < 0.7:
        if 'fluency_issue' in fluency_details:
            feedback_parts.append(f"⚠️ Argumentative Fluency: {fluency_details['fluency_issue']}. "
                                f"Found {fluency_details['total_connectors']} connectors "
                                f"({fluency_details['connector_density']:.1f} per 1000 words).")
        if 'diversity_issue' in fluency_details:
            feedback_parts.append(f"Consider using more diverse logical connectors. {fluency_details['diversity_issue']}.")
    
    # Connectivity feedback
    if connectivity_score < 0.7:
        if 'connectivity_issue' in connectivity_details:
            feedback_parts.append(f"⚠️ Section Connectivity: {connectivity_details['connectivity_issue']}. "
                                f"Add explicit cross-references between sections.")
        if connectivity_details.get('reference_count', 0) == 0:
            feedback_parts.append("Consider adding phrases like 'as discussed in' or 'see Section X'.")
    
    # Consistency feedback
    if consistency_score < 0.7:
        if 'consistency_issue' in consistency_details:
            feedback_parts.append(f"⚠️ Narrative Consistency: {consistency_details['consistency_issue']}.")
        if 'pov_issue' in consistency_details:
            feedback_parts.append(f"{consistency_details['pov_issue']}. "
                                f"Maintain consistent first-person ('we') or third-person ('this study') throughout.")
        if consistency_details.get('terminology_inconsistencies', 0) > 0:
            feedback_parts.append(f"Found {consistency_details['terminology_inconsistencies']} "
                                f"terminology variations. Ensure consistent use of technical terms.")
    
    # Flow feedback
    if flow_score < 0.7:
        if 'flow_issue' in flow_details:
            feedback_parts.append(f"⚠️ Logical Flow: {flow_details['flow_issue']}. "
                                f"Follow the standard structure: Introduction → Methodology → Results → Discussion → Conclusion.")
        if flow_details.get('conclusion_markers', 0) == 0:
            feedback_parts.append("Add clear conclusion markers (e.g., 'In conclusion', 'To summarize').")
        if not flow_details.get('intro_conclusion_link', True):
            feedback_parts.append("Strengthen the link between introduction and conclusion by referring back to initial objectives.")
    
    # Positive feedback
    if not feedback_parts:
        strengths = []
        if fluency_score >= 0.8:
            strengths.append("excellent use of logical connectors")
        if connectivity_score >= 0.8:
            strengths.append("strong cross-references between sections")
        if consistency_score >= 0.8:
            strengths.append("consistent terminology and narrative voice")
        if flow_score >= 0.8:
            strengths.append("logical section progression")
        
        if strengths:
            feedback_parts.append(f"✓ Strong coherence and cohesion with {', '.join(strengths)}.")
        else:
            feedback_parts.append("✓ Good overall coherence and cohesion.")
    
    return " ".join(feedback_parts)