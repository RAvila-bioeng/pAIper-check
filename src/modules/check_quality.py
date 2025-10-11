"""
Scientific Quality Evaluation Module
The core of the review, focusing on contribution and intrinsic rigor.
"""

import re
from typing import List, Tuple
from models.score import PillarResult


def evaluate(paper) -> dict:
    """
    Evaluate scientific quality of the paper.
    
    Args:
        paper: Paper object with text content
        
    Returns:
        dict: Score and feedback for scientific quality
    """
    text = paper.full_text
    
    # Check novelty and originality
    novelty_score = _check_novelty_originality(text)
    
    # Check methodological rigor
    rigor_score = _check_methodological_rigor(text, paper.sections)
    
    # Check significance of results
    significance_score = _check_results_significance(text)
    
    # Check theoretical contribution
    theory_score = _check_theoretical_contribution(text)
    
    # Check practical implications
    practical_score = _check_practical_implications(text)
    
    # Calculate overall score
    overall_score = (novelty_score + rigor_score + significance_score + theory_score + practical_score) / 5
    
    # Generate feedback
    feedback = _generate_quality_feedback(novelty_score, rigor_score, significance_score, theory_score, practical_score)
    
    return PillarResult("Scientific Quality", overall_score, feedback).__dict__


def _check_novelty_originality(text: str) -> float:
    """Analyze the paper's contribution to the existing body of knowledge."""
    score = 1.0
    
    # Check for novelty indicators
    novelty_indicators = [
        'novel', 'new', 'innovative', 'original', 'first', 'propose',
        'introduce', 'develop', 'create', 'design', 'invent',
        'breakthrough', 'advancement', 'improvement', 'enhancement'
    ]
    
    novelty_count = sum(len(re.findall(rf'\b{indicator}\b', text.lower())) for indicator in novelty_indicators)
    
    if novelty_count < 3:
        score -= 0.4
    elif novelty_count < 5:
        score -= 0.2
    
    # Check for contribution statements
    contribution_phrases = [
        'main contribution', 'key contribution', 'primary contribution',
        'this paper contributes', 'we contribute', 'our contribution',
        'the contribution of this work', 'significant contribution'
    ]
    
    contribution_count = sum(1 for phrase in contribution_phrases if phrase in text.lower())
    
    if contribution_count == 0:
        score -= 0.3
    elif contribution_count > 1:
        score += 0.1
    
    # Check for problem statement and motivation
    problem_indicators = [
        'problem', 'challenge', 'issue', 'limitation', 'gap',
        'motivation', 'motivated by', 'inspired by', 'addresses'
    ]
    
    problem_count = sum(1 for indicator in problem_indicators if indicator in text.lower())
    
    if problem_count < 2:
        score -= 0.2
    
    # Check for comparison with existing work
    comparison_phrases = [
        'compared to', 'in contrast to', 'unlike', 'different from',
        'improves upon', 'extends', 'builds on', 'advances'
    ]
    
    comparison_count = sum(1 for phrase in comparison_phrases if phrase in text.lower())
    
    if comparison_count < 2:
        score -= 0.2
    
    return max(0.0, score)


def _check_methodological_rigor(text: str, sections: List) -> float:
    """Evaluate the suitability of the selected methods for the stated objectives."""
    score = 1.0
    
    # Check for clear objectives
    objective_phrases = [
        'objective', 'goal', 'aim', 'purpose', 'intend to',
        'seek to', 'attempt to', 'strive to'
    ]
    
    objective_count = sum(1 for phrase in objective_phrases if phrase in text.lower())
    
    if objective_count == 0:
        score -= 0.3
    elif objective_count > 2:
        score += 0.1
    
    # Check for hypothesis or research questions
    hypothesis_indicators = [
        'hypothesis', 'research question', 'research problem',
        'we hypothesize', 'we propose that', 'we expect'
    ]
    
    hypothesis_count = sum(1 for indicator in hypothesis_indicators if indicator in text.lower())
    
    if hypothesis_count == 0:
        score -= 0.2
    
    # Check for appropriate methodology
    method_indicators = [
        'experiment', 'study', 'analysis', 'evaluation', 'assessment',
        'validation', 'verification', 'testing', 'simulation'
    ]
    
    method_count = sum(1 for indicator in method_indicators if indicator in text.lower())
    
    if method_count < 2:
        score -= 0.3
    
    # Check for statistical analysis
    statistical_indicators = [
        'statistical', 'significance', 'p-value', 'confidence interval',
        'correlation', 'regression', 'anova', 't-test', 'chi-square'
    ]
    
    statistical_count = sum(1 for indicator in statistical_indicators if indicator in text.lower())
    
    if statistical_count > 0:
        score += 0.1
    
    # Check for experimental design
    design_indicators = [
        'experimental design', 'study design', 'research design',
        'control group', 'treatment group', 'randomized', 'blinded'
    ]
    
    design_count = sum(1 for indicator in design_indicators if indicator in text.lower())
    
    if design_count > 0:
        score += 0.1
    
    # Check for limitations acknowledgment
    limitation_phrases = [
        'limitation', 'limitation of this study', 'constraint',
        'drawback', 'weakness', 'shortcoming'
    ]
    
    limitation_count = sum(1 for phrase in limitation_phrases if phrase in text.lower())
    
    if limitation_count > 0:
        score += 0.1  # Acknowledging limitations is good practice
    
    return min(1.0, max(0.0, score))


def _check_results_significance(text: str) -> float:
    """Validate the importance and implications of the findings."""
    score = 1.0
    
    # Check for results section
    results_section = None
    if 'results' in text.lower():
        # Find results section content
        results_match = re.search(r'results\s*[:\-]?\s*(.*?)(?=discussion|conclusion|$)', text, re.DOTALL | re.IGNORECASE)
        if results_match:
            results_section = results_match.group(1).lower()
    
    if not results_section:
        return 0.4  # No clear results section
    
    # Check for quantitative results
    quantitative_indicators = [
        r'\d+\.?\d*%',  # Percentages
        r'\d+\.?\d*\s*(increase|decrease|improvement|reduction)',
        r'\d+\.?\d*\s*(times|fold)',  # Multiples
        r'significantly\s+(higher|lower|better|worse)',
        r'p\s*[<>=]\s*0\.\d+',  # P-values
        r'confidence\s+interval'
    ]
    
    quantitative_count = sum(len(re.findall(pattern, results_section)) for pattern in quantitative_indicators)
    
    if quantitative_count < 3:
        score -= 0.3
    elif quantitative_count > 5:
        score += 0.1
    
    # Check for significance statements
    significance_phrases = [
        'significant', 'significantly', 'statistically significant',
        'important', 'meaningful', 'substantial', 'considerable',
        'noteworthy', 'remarkable', 'striking'
    ]
    
    significance_count = sum(1 for phrase in significance_phrases if phrase in results_section)
    
    if significance_count < 2:
        score -= 0.2
    
    # Check for comparison with baselines or benchmarks
    comparison_phrases = [
        'compared to', 'versus', 'vs', 'in comparison',
        'baseline', 'benchmark', 'state-of-the-art',
        'previous work', 'existing methods'
    ]
    
    comparison_count = sum(1 for phrase in comparison_phrases if phrase in results_section)
    
    if comparison_count < 1:
        score -= 0.3
    elif comparison_count > 3:
        score += 0.1
    
    return max(0.0, score)


def _check_theoretical_contribution(text: str) -> float:
    """Check for theoretical contributions and implications."""
    score = 1.0
    
    # Check for theoretical framework
    theory_indicators = [
        'theoretical', 'theory', 'framework', 'model', 'concept',
        'principle', 'assumption', 'postulate', 'hypothesis'
    ]
    
    theory_count = sum(1 for indicator in theory_indicators if indicator in text.lower())
    
    if theory_count < 3:
        score -= 0.3
    
    # Check for conceptual contributions
    conceptual_phrases = [
        'conceptual', 'conceptually', 'theoretically', 'in theory',
        'theoretical framework', 'conceptual framework'
    ]
    
    conceptual_count = sum(1 for phrase in conceptual_phrases if phrase in text.lower())
    
    if conceptual_count > 0:
        score += 0.1
    
    # Check for model or framework development
    development_phrases = [
        'propose a model', 'develop a framework', 'introduce a concept',
        'theoretical model', 'conceptual model', 'analytical framework'
    ]
    
    development_count = sum(1 for phrase in development_phrases if phrase in text.lower())
    
    if development_count > 0:
        score += 0.2
    
    # Check for implications for theory
    implication_phrases = [
        'implications for theory', 'theoretical implications',
        'contributes to theory', 'advances theory'
    ]
    
    implication_count = sum(1 for phrase in implication_phrases if phrase in text.lower())
    
    if implication_count > 0:
        score += 0.1
    
    return min(1.0, max(0.0, score))


def _check_practical_implications(text: str) -> float:
    """Check for practical applications and implications."""
    score = 1.0
    
    # Check for practical applications
    practical_indicators = [
        'practical', 'application', 'applied', 'implement',
        'implementation', 'deploy', 'deployment', 'use case',
        'real-world', 'real world', 'industry', 'commercial'
    ]
    
    practical_count = sum(1 for indicator in practical_indicators if indicator in text.lower())
    
    if practical_count < 2:
        score -= 0.3
    
    # Check for future work or applications
    future_phrases = [
        'future work', 'future research', 'future applications',
        'potential applications', 'possible applications',
        'could be applied', 'can be used', 'useful for'
    ]
    
    future_count = sum(1 for phrase in future_phrases if phrase in text.lower())
    
    if future_count > 0:
        score += 0.1
    
    # Check for societal or broader impact
    impact_phrases = [
        'impact', 'implications', 'significance', 'importance',
        'benefit', 'contribution to society', 'broader impact'
    ]
    
    impact_count = sum(1 for phrase in impact_phrases if phrase in text.lower())
    
    if impact_count < 2:
        score -= 0.2
    
    # Check for limitations and future directions
    limitation_phrases = [
        'limitation', 'future work', 'future research', 'next steps',
        'further research', 'additional work', 'improvements'
    ]
    
    limitation_count = sum(1 for phrase in limitation_phrases if phrase in text.lower())
    
    if limitation_count > 0:
        score += 0.1  # Good practice to acknowledge limitations
    
    return min(1.0, max(0.0, score))


def _generate_quality_feedback(novelty_score: float, rigor_score: float, 
                             significance_score: float, theory_score: float, 
                             practical_score: float) -> str:
    """Generate detailed feedback for scientific quality."""
    feedback_parts = []
    
    if novelty_score < 0.7:
        feedback_parts.append("Novelty and originality need improvement. Clearly state your contributions and how they advance the field.")
    
    if rigor_score < 0.7:
        feedback_parts.append("Methodological rigor could be enhanced. Provide clearer objectives, hypotheses, and experimental design.")
    
    if significance_score < 0.7:
        feedback_parts.append("Results significance needs better demonstration. Include more quantitative results and comparisons with baselines.")
    
    if theory_score < 0.7:
        feedback_parts.append("Theoretical contribution could be stronger. Develop clearer theoretical framework and implications.")
    
    if practical_score < 0.7:
        feedback_parts.append("Practical implications need better articulation. Discuss applications and broader impact more clearly.")
    
    if not feedback_parts:
        feedback_parts.append("Strong scientific quality with clear contributions, rigorous methodology, and significant results.")
    
    return " ".join(feedback_parts)
