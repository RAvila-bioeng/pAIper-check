"""
Reproducibility Assessment Module
A crucial pillar for modern research, centered on methodological transparency.
"""

import re
from typing import List, Tuple
from models.score import PillarResult


def evaluate(paper) -> dict:
    """
    Evaluate reproducibility aspects of the paper.
    
    Args:
        paper: Paper object with text content
        
    Returns:
        dict: Score and feedback for reproducibility
    """
    text = paper.full_text.lower()
    
    # Check methodological clarity
    clarity_score = _check_methodological_clarity(text, paper.sections)
    
    # Check data availability
    data_score = _check_data_availability(text)
    
    # Check code availability
    code_score = _check_code_availability(text)
    
    # Check parameter specification
    parameter_score = _check_parameter_specification(text)
    
    # Check replicability indicators
    replicability_score = _check_replicability_indicators(text)
    
    # Calculate overall score
    overall_score = (clarity_score + data_score + code_score + parameter_score + replicability_score) / 5
    
    # Generate feedback
    feedback = _generate_reproducibility_feedback(clarity_score, data_score, code_score, parameter_score, replicability_score)
    
    return PillarResult("Reproducibility", overall_score, feedback).__dict__


def _check_methodological_clarity(text: str, sections: List) -> float:
    """Analyze the sufficiency of method descriptions to allow replication."""
    score = 1.0
    
    # Check for detailed methodology section
    method_section = None
    for section in sections:
        if 'method' in section.title.lower():
            method_section = section.content.lower()
            break
    
    if not method_section:
        return 0.3  # No methodology section
    
    # Check for specific methodological details
    methodological_indicators = [
        'algorithm', 'procedure', 'protocol', 'workflow', 'pipeline',
        'implementation', 'configuration', 'setup', 'environment',
        'hardware', 'software', 'framework', 'library', 'tool'
    ]
    
    indicator_count = sum(1 for indicator in methodological_indicators if indicator in method_section)
    
    if indicator_count < 3:
        score -= 0.4
    elif indicator_count < 5:
        score -= 0.2
    
    # Check for step-by-step descriptions
    step_indicators = [
        r'step\s+\d+', r'first\s*,?\s*second\s*,?\s*third',
        r'initially\s*,?\s*then\s*,?\s*finally',
        r'begin\s+by\s*,?\s*next\s*,?\s*after'
    ]
    
    step_count = sum(len(re.findall(pattern, method_section)) for pattern in step_indicators)
    
    if step_count == 0:
        score -= 0.3
    elif step_count < 3:
        score -= 0.1
    
    # Check for experimental design details
    design_indicators = [
        'control', 'baseline', 'comparison', 'evaluation', 'metric',
        'dataset', 'training', 'testing', 'validation', 'split'
    ]
    
    design_count = sum(1 for indicator in design_indicators if indicator in method_section)
    
    if design_count < 3:
        score -= 0.2
    
    return max(0.0, score)


def _check_data_availability(text: str) -> float:
    """Verification of mentions regarding accessibility to data."""
    score = 0.0
    
    # Check for data availability statements
    availability_phrases = [
        'data is available', 'dataset is available', 'data can be accessed',
        'data is provided', 'supplementary data', 'data repository',
        'github', 'figshare', 'zenodo', 'dataverse', 'dryad',
        'upon request', 'contact the authors', 'data available on request'
    ]
    
    availability_count = sum(1 for phrase in availability_phrases if phrase in text)
    
    if availability_count > 0:
        score = 0.8
        if any(repo in text for repo in ['github', 'figshare', 'zenodo', 'dataverse', 'dryad']):
            score = 1.0  # Public repository mentioned
    
    # Check for data description
    description_phrases = [
        'dataset contains', 'data consists of', 'sample size',
        'number of samples', 'data points', 'records',
        'features', 'variables', 'dimensions'
    ]
    
    description_count = sum(1 for phrase in description_phrases if phrase in text)
    
    if description_count > 0:
        score += 0.1
    
    return min(1.0, score)


def _check_code_availability(text: str) -> float:
    """Check for code availability and sharing."""
    score = 0.0
    
    # Check for code availability statements
    code_phrases = [
        'code is available', 'source code', 'implementation is available',
        'github repository', 'code repository', 'software is available',
        'algorithm implementation', 'code can be accessed'
    ]
    
    code_count = sum(1 for phrase in code_phrases if phrase in text)
    
    if code_count > 0:
        score = 0.8
        if 'github' in text:
            score = 1.0  # GitHub repository mentioned
    
    # Check for programming language mentions (indicates code existence)
    languages = ['python', 'r', 'matlab', 'java', 'c++', 'javascript', 'ruby']
    language_count = sum(1 for lang in languages if lang in text)
    
    if language_count > 0:
        score += 0.2
    
    return min(1.0, score)


def _check_parameter_specification(text: str) -> float:
    """Check for detailed parameter specifications."""
    score = 1.0
    
    # Check for hyperparameter mentions
    hyperparameter_indicators = [
        'learning rate', 'batch size', 'epoch', 'parameter', 'hyperparameter',
        'threshold', 'window size', 'kernel size', 'regularization',
        'dropout', 'momentum', 'optimizer', 'activation function'
    ]
    
    param_count = sum(1 for indicator in hyperparameter_indicators if indicator in text)
    
    if param_count == 0:
        score = 0.3
    elif param_count < 3:
        score -= 0.3
    elif param_count < 5:
        score -= 0.1
    
    # Check for numerical values (indicating specific parameters)
    numerical_values = re.findall(r'\b\d+\.?\d*\b', text)
    unique_values = len(set(numerical_values))
    
    if unique_values < 5:
        score -= 0.2
    
    # Check for configuration details
    config_indicators = [
        'configuration', 'settings', 'parameters', 'default values',
        'experimental setup', 'system configuration'
    ]
    
    config_count = sum(1 for indicator in config_indicators if indicator in text)
    
    if config_count == 0:
        score -= 0.2
    
    return max(0.0, score)


def _check_replicability_indicators(text: str) -> float:
    """Evaluate theoretical possibility of reproducing experiments or analysis."""
    score = 1.0
    
    # Check for reproducibility statements
    repro_phrases = [
        'reproducible', 'replication', 'replicable', 'reproduce',
        'reproducibility', 'replication study', 'can be reproduced'
    ]
    
    repro_count = sum(1 for phrase in repro_phrases if phrase in text)
    
    if repro_count == 0:
        score -= 0.3
    
    # Check for limitations and constraints
    limitation_phrases = [
        'limitation', 'constraint', 'restriction', 'limitation of this study',
        'future work', 'improvement', 'enhancement'
    ]
    
    limitation_count = sum(1 for phrase in limitation_phrases if phrase in text)
    
    if limitation_count > 0:
        score += 0.1  # Acknowledging limitations is good for reproducibility
    
    # Check for validation and verification
    validation_phrases = [
        'validation', 'verification', 'confirmed', 'validated',
        'cross-validation', 'independent validation'
    ]
    
    validation_count = sum(1 for phrase in validation_phrases if phrase in text)
    
    if validation_count == 0:
        score -= 0.2
    
    # Check for baseline comparisons
    baseline_phrases = [
        'baseline', 'comparison', 'compared to', 'state-of-the-art',
        'existing methods', 'previous work'
    ]
    
    baseline_count = sum(1 for phrase in baseline_phrases if phrase in text)
    
    if baseline_count < 2:
        score -= 0.2
    
    return max(0.0, score)


def _generate_reproducibility_feedback(clarity_score: float, data_score: float, 
                                     code_score: float, parameter_score: float, 
                                     replicability_score: float) -> str:
    """Generate detailed feedback for reproducibility."""
    feedback_parts = []
    
    if clarity_score < 0.7:
        feedback_parts.append("Methodology description lacks sufficient detail for replication. Provide more step-by-step procedures.")
    
    if data_score < 0.5:
        feedback_parts.append("Data availability information is missing or unclear. Specify how data can be accessed.")
    
    if code_score < 0.5:
        feedback_parts.append("Code availability is not mentioned. Consider sharing implementation details or code repository.")
    
    if parameter_score < 0.7:
        feedback_parts.append("Parameter specifications are insufficient. Provide detailed hyperparameter values and configurations.")
    
    if replicability_score < 0.7:
        feedback_parts.append("Reproducibility aspects need improvement. Add validation details and baseline comparisons.")
    
    if not feedback_parts:
        feedback_parts.append("Good reproducibility practices with clear methodology, data access, and validation details.")
    
    return " ".join(feedback_parts)
