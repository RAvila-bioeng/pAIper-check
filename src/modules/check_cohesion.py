"""
Coherence and Cohesion Analysis Module
Evaluates the flow and logical connection of the argument.
"""

import re
from typing import List, Tuple
from models.score import PillarResult


def evaluate(paper) -> dict:
    """
    Evaluate coherence and cohesion of the paper.
    
    Args:
        paper: Paper object with text content
        
    Returns:
        dict: Score and feedback for coherence and cohesion
    """
    text = paper.full_text
    
    # Check argumentative fluency
    fluency_score = _check_argumentative_fluency(text)
    
    # Check connectivity between sections
    connectivity_score = _check_section_connectivity(text, paper.sections)
    
    # Check narrative consistency
    consistency_score = _check_narrative_consistency(text)
    
    # Check logical flow
    flow_score = _check_logical_flow(text)
    
    # Calculate overall score
    overall_score = (fluency_score + connectivity_score + consistency_score + flow_score) / 4
    
    # Generate feedback
    feedback = _generate_cohesion_feedback(fluency_score, connectivity_score, consistency_score, flow_score)
    
    return PillarResult("Coherence & Cohesion", overall_score, feedback).__dict__


def _check_argumentative_fluency(text: str) -> float:
    """Check how well ideas are developed throughout the manuscript."""
    score = 1.0
    
    # Check for logical connectors
    connectors = [
        'therefore', 'thus', 'hence', 'consequently', 'as a result',
        'however', 'nevertheless', 'on the other hand', 'in contrast',
        'furthermore', 'moreover', 'additionally', 'in addition',
        'firstly', 'secondly', 'finally', 'in conclusion'
    ]
    
    connector_count = sum(len(re.findall(rf'\b{connector}\b', text.lower())) for connector in connectors)
    
    # Good papers should have multiple logical connectors
    if connector_count < 5:
        score -= 0.3
    elif connector_count < 10:
        score -= 0.1
    
    # Check for proper use of transitions between paragraphs
    paragraph_transitions = len(re.findall(r'\.\s*\n\s*[A-Z][^.]*?(however|furthermore|moreover|additionally|in addition|therefore|thus|consequently)', text, re.IGNORECASE))
    
    if paragraph_transitions < 3:
        score -= 0.2
    
    return max(0.0, score)


def _check_section_connectivity(text: str, sections: List) -> float:
    """Check logical relationship between paragraphs and sections."""
    if not sections:
        return 0.5  # Can't evaluate without sections
    
    score = 1.0
    
    # Check if sections reference each other appropriately
    section_titles = [section.title.lower() for section in sections]
    
    # Look for cross-references between sections
    cross_references = 0
    for i, title in enumerate(section_titles):
        for j, other_title in enumerate(section_titles):
            if i != j and other_title in text.lower():
                cross_references += 1
    
    if cross_references < len(sections) * 0.5:
        score -= 0.3
    
    # Check for section summaries and transitions
    transition_phrases = [
        'as discussed in', 'as mentioned previously', 'in the previous section',
        'building on', 'extending the analysis', 'following the methodology'
    ]
    
    transition_count = sum(len(re.findall(rf'\b{phrase}\b', text.lower())) for phrase in transition_phrases)
    
    if transition_count < 2:
        score -= 0.2
    
    return max(0.0, score)


def _check_narrative_consistency(text: str) -> float:
    """Ensure uniform storyline from introduction to conclusions."""
    score = 1.0
    
    # Check for consistent terminology throughout
    # Look for key terms that should appear consistently
    key_terms = re.findall(r'\b[a-z]{4,}\b', text.lower())
    term_frequency = {}
    
    for term in key_terms:
        if len(term) > 4 and term not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'could', 'there', 'after', 'first', 'well', 'also', 'where', 'much', 'some', 'very', 'when', 'make', 'more', 'over', 'think', 'help', 'good', 'same', 'many', 'most', 'other', 'new', 'way', 'may', 'say', 'use', 'man', 'find', 'give', 'year', 'work', 'part', 'take', 'get', 'place', 'made', 'live', 'where', 'through', 'back', 'much', 'before', 'line', 'right', 'too', 'means', 'old', 'any', 'same', 'tell', 'boy', 'follow', 'came', 'want', 'show', 'also', 'around', 'form', 'three', 'small', 'set', 'put', 'end', 'why', 'again', 'turn', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were']:
            term_frequency[term] = term_frequency.get(term, 0) + 1
    
    # Check for terms that appear frequently (should be consistent)
    frequent_terms = [term for term, count in term_frequency.items() if count > 3]
    
    # Look for variations of frequent terms that might indicate inconsistency
    inconsistency_count = 0
    for term in frequent_terms:
        # Look for variations (e.g., "method" vs "methodology")
        variations = []
        if term.endswith('s'):
            variations.append(term[:-1])
        if not term.endswith('y'):
            variations.append(term + 'y')
        if not term.endswith('ion'):
            variations.append(term + 'ion')
        
        for variation in variations:
            if variation in term_frequency and term_frequency[variation] > 1:
                inconsistency_count += 1
    
    if inconsistency_count > len(frequent_terms) * 0.3:
        score -= 0.3
    
    # Check for consistent point of view (first person vs third person)
    first_person_count = len(re.findall(r'\b(we|our|us|ourselves)\b', text.lower()))
    third_person_count = len(re.findall(r'\b(this study|this research|this paper|the authors)\b', text.lower()))
    
    if first_person_count > 0 and third_person_count > 0:
        # Mixed point of view
        if abs(first_person_count - third_person_count) > max(first_person_count, third_person_count) * 0.5:
            score -= 0.2
    
    return max(0.0, score)


def _check_logical_flow(text: str) -> float:
    """Check the logical progression of ideas."""
    score = 1.0
    
    # Check for proper introduction of concepts before use
    # This is a simplified check - in practice, this would need more sophisticated NLP
    
    # Check for "methodology -> results -> discussion" flow
    sections = ['method', 'result', 'discussion', 'conclusion']
    section_order = []
    
    for section in sections:
        if section in text.lower():
            # Find approximate position in text
            position = text.lower().find(section)
            section_order.append((section, position))
    
    # Sort by position
    section_order.sort(key=lambda x: x[1])
    
    # Check if order is logical (method before results, results before discussion, etc.)
    expected_order = ['method', 'result', 'discussion', 'conclusion']
    found_sections = [section for section, _ in section_order]
    
    logical_flow_score = 0
    for i, section in enumerate(found_sections):
        if section in expected_order:
            expected_index = expected_order.index(section)
            if i == expected_index or (i > 0 and found_sections[i-1] in expected_order and expected_order.index(found_sections[i-1]) < expected_index):
                logical_flow_score += 1
    
    if len(found_sections) > 1:
        flow_ratio = logical_flow_score / len(found_sections)
        if flow_ratio < 0.7:
            score -= 0.3
    
    # Check for proper conclusion that ties back to introduction
    conclusion_phrases = ['in conclusion', 'to summarize', 'in summary', 'finally', 'overall']
    conclusion_count = sum(len(re.findall(rf'\b{phrase}\b', text.lower())) for phrase in conclusion_phrases)
    
    if conclusion_count == 0:
        score -= 0.2
    
    return max(0.0, score)


def _generate_cohesion_feedback(fluency_score: float, connectivity_score: float, 
                               consistency_score: float, flow_score: float) -> str:
    """Generate detailed feedback for coherence and cohesion."""
    feedback_parts = []
    
    if fluency_score < 0.7:
        feedback_parts.append("Limited use of logical connectors. Improve transitions between ideas.")
    
    if connectivity_score < 0.7:
        feedback_parts.append("Sections lack proper connections. Add more cross-references and transitions.")
    
    if consistency_score < 0.7:
        feedback_parts.append("Inconsistent terminology or narrative voice. Ensure consistent language throughout.")
    
    if flow_score < 0.7:
        feedback_parts.append("Logical flow could be improved. Ensure proper progression from methodology to results to discussion.")
    
    if not feedback_parts:
        feedback_parts.append("Good coherence and cohesion with logical flow and consistent narrative.")
    
    return " ".join(feedback_parts)
