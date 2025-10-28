"""
Scientific Quality Evaluation Module
The core of the review, focusing on contribution and intrinsic rigor.
Enhanced with advanced metrics and optional GPT integration for deeper analysis.
"""

import re
import os
import json
import time
from typing import List, Tuple, Dict, Optional
from models.score import PillarResult

from modules.gpt_quality_analyzer import enhance_quality_with_gpt, GPT_AVAILABLE


def evaluate(paper, use_gpt: bool = False) -> dict:
    """
    Evaluate scientific quality of the paper with enhanced metrics.
    
    Args:
        paper: Paper object with text content
        use_gpt: Whether to use GPT for deeper analysis
        
    Returns:
        dict: Score and feedback for scientific quality
    """
    text = paper.full_text
    
    # Enhanced novelty and originality analysis
    novelty_score, novelty_details = _check_novelty_originality_enhanced(text, paper)
    
    # Enhanced methodological rigor analysis
    rigor_score, rigor_details = _check_methodological_rigor_enhanced(text, paper.sections)
    
    # Enhanced significance of results analysis
    significance_score, significance_details = _check_results_significance_enhanced(text, paper)
    
    # Enhanced theoretical contribution analysis
    theory_score, theory_details = _check_theoretical_contribution_enhanced(text, paper)
    
    # Enhanced practical implications analysis
    practical_score, practical_details = _check_practical_implications_enhanced(text, paper)
    
    # Advanced metrics
    impact_score, impact_details = _check_research_impact(text, paper)
    reproducibility_score, reproducibility_details = _check_reproducibility_quality(text, paper)
    
    # Calculate weighted overall score
    weights = {
        'novelty': 0.25,
        'rigor': 0.20,
        'significance': 0.20,
        'theory': 0.15,
        'practical': 0.10,
        'impact': 0.05,
        'reproducibility': 0.05
    }
    
    overall_score = (
        novelty_score * weights['novelty'] +
        rigor_score * weights['rigor'] +
        significance_score * weights['significance'] +
        theory_score * weights['theory'] +
        practical_score * weights['practical'] +
        impact_score * weights['impact'] +
        reproducibility_score * weights['reproducibility']
    )
    
    # Generate detailed feedback
    feedback = _generate_quality_feedback_enhanced(
        novelty_score, novelty_details,
        rigor_score, rigor_details,
        significance_score, significance_details,
        theory_score, theory_details,
        practical_score, practical_details,
        impact_score, impact_details,
        reproducibility_score, reproducibility_details
    )
    
    basic_result = PillarResult("Scientific Quality", overall_score, feedback).__dict__
    
    # Add detailed breakdown and analysis
    basic_result['score_breakdown'] = {
        'novelty': novelty_score,
        'rigor': rigor_score,
        'significance': significance_score,
        'theory': theory_score,
        'practical': practical_score,
        'impact': impact_score,
        'reproducibility': reproducibility_score
    }

    basic_result['detailed_analysis'] = {
        'novelty': novelty_details,
        'rigor': rigor_details,
        'significance': significance_details,
        'theory': theory_details,
        'practical': practical_details,
        'impact': impact_details,
        'reproducibility': reproducibility_details
    }

    # ✅ CORRECCIÓN: GPT enhancement - pasar basic_result completo
    if use_gpt and GPT_AVAILABLE:
        # La función enhance_quality_with_gpt modifica basic_result directamente
        # y agrega el campo 'gpt_analysis' internamente
        basic_result = enhance_quality_with_gpt(paper, basic_result, force_analysis=False)
    else:
        # Si no hay GPT disponible, añadir info de por qué no se usó
        if use_gpt and not GPT_AVAILABLE:
            basic_result['gpt_analysis'] = {
                'used': False,
                'success': False,
                'reason': 'GPT dependencies not installed',
                'error': 'The "openai" package is not installed. Please install it with "pip install openai".'
            }

    return basic_result


def _check_novelty_originality_enhanced(text: str, paper) -> Tuple[float, Dict]:
    """Enhanced analysis of the paper's contribution to the existing body of knowledge."""
    score = 1.0
    details = {
        'novelty_indicators_found': 0,
        'contribution_statements': 0,
        'problem_statements': 0,
        'comparisons': 0,
        'specific_contributions': [],
        'gaps_identified': [],
        'improvements_claimed': []
    }
    
    # Enhanced novelty indicators with context awareness
    novelty_patterns = [
        (r'\b(novel|new|innovative|original|first|pioneering)\b', 'novelty'),
        (r'\b(propose|introduce|develop|create|design|invent)\b', 'creation'),
        (r'\b(breakthrough|advancement|improvement|enhancement|revolutionary)\b', 'advancement'),
        (r'\b(extend|expand|build upon|advance|refine)\b', 'extension')
    ]
    
    novelty_count = 0
    for pattern, category in novelty_patterns:
        matches = re.findall(pattern, text.lower())
        novelty_count += len(matches)
        if category == 'novelty' and matches:
            details['specific_contributions'].extend(matches[:3])  # Store first 3 examples
    
    details['novelty_indicators_found'] = novelty_count
    
    # Score based on novelty density (normalized by text length)
    text_length = len(text.split())
    novelty_density = novelty_count / max(text_length / 1000, 1)  # per 1000 words
    
    if novelty_density < 0.5:
        score -= 0.4
    elif novelty_density < 1.0:
        score -= 0.2
    elif novelty_density > 2.0:
        score += 0.1
    
    # Enhanced contribution statements analysis
    contribution_patterns = [
        r'main contribution[s]?',
        r'key contribution[s]?',
        r'primary contribution[s]?',
        r'this paper contributes?',
        r'we contribute?',
        r'our contribution[s]?',
        r'the contribution[s]? of this work',
        r'significant contribution[s]?',
        r'original contribution[s]?'
    ]
    
    contribution_count = 0
    for pattern in contribution_patterns:
        matches = re.findall(pattern, text.lower())
        contribution_count += len(matches)
    
    details['contribution_statements'] = contribution_count
    
    if contribution_count == 0:
        score -= 0.3
    elif contribution_count > 2:
        score += 0.1
    
    # Enhanced problem statement analysis
    problem_patterns = [
        r'\b(problem|challenge|issue|limitation|gap|shortcoming)\b',
        r'\b(motivation|motivated by|inspired by|addresses?)\b',
        r'\b(need for|requires?|necessitates?)\b',
        r'\b(currently|existing|previous work)\b.*\b(limitation|problem|issue)\b'
    ]
    
    problem_count = 0
    for pattern in problem_patterns:
        matches = re.findall(pattern, text.lower())
        problem_count += len(matches)
    
    details['problem_statements'] = problem_count
    
    if problem_count < 3:
        score -= 0.2
    elif problem_count > 5:
        score += 0.1
    
    # Enhanced comparison analysis
    comparison_patterns = [
        r'\b(compared to|in contrast to|unlike|different from)\b',
        r'\b(improves? upon|extends?|builds? on|advances?)\b',
        r'\b(superior to|better than|outperforms?)\b',
        r'\b(state.?of.?the.?art|baseline|benchmark)\b'
    ]
    
    comparison_count = 0
    for pattern in comparison_patterns:
        matches = re.findall(pattern, text.lower())
        comparison_count += len(matches)
    
    details['comparisons'] = comparison_count
    
    if comparison_count < 2:
        score -= 0.2
    elif comparison_count > 4:
        score += 0.1
    
    # Check for specific research gaps
    gap_patterns = [
        r'gap in.*research',
        r'limited research on',
        r'lack of.*studies',
        r'few studies have',
        r'no previous work'
    ]
    
    gaps_found = []
    for pattern in gap_patterns:
        matches = re.findall(pattern, text.lower())
        gaps_found.extend(matches)
    
    details['gaps_identified'] = gaps_found[:3]  # Store first 3 examples
    
    if gaps_found:
        score += 0.1
    
    return min(1.0, max(0.0, score)), details


def _check_methodological_rigor_enhanced(text: str, sections: List) -> Tuple[float, Dict]:
    """Enhanced evaluation of the suitability of the selected methods for the stated objectives."""
    score = 1.0
    details = {
        'objectives_clarity': 0,
        'hypothesis_presence': 0,
        'method_appropriateness': 0,
        'statistical_analysis': 0,
        'experimental_design': 0,
        'limitations_acknowledged': 0,
        'method_details': [],
        'statistical_methods': [],
        'design_elements': []
    }
    
    # Enhanced objectives analysis
    objective_patterns = [
        r'\b(objective[s]?|goal[s]?|aim[s]?|purpose[s]?)\b',
        r'\b(intend[s]? to|seek[s]? to|attempt[s]? to|strive[s]? to)\b',
        r'\b(investigate|examine|explore|analyze|evaluate)\b',
        r'\b(research question[s]?|research problem[s]?)\b'
    ]
    
    objective_count = 0
    for pattern in objective_patterns:
        matches = re.findall(pattern, text.lower())
        objective_count += len(matches)
    
    details['objectives_clarity'] = objective_count
    
    if objective_count == 0:
        score -= 0.3
    elif objective_count > 3:
        score += 0.1
    
    # Enhanced hypothesis analysis
    hypothesis_patterns = [
        r'\b(hypothesis|hypotheses)\b',
        r'\b(we hypothesize|we propose that|we expect)\b',
        r'\b(null hypothesis|alternative hypothesis)\b',
        r'\b(predict[s]? that|anticipate[s]? that)\b'
    ]
    
    hypothesis_count = 0
    for pattern in hypothesis_patterns:
        matches = re.findall(pattern, text.lower())
        hypothesis_count += len(matches)
    
    details['hypothesis_presence'] = hypothesis_count
    
    if hypothesis_count == 0:
        score -= 0.2
    elif hypothesis_count > 1:
        score += 0.1
    
    # Enhanced methodology analysis
    method_patterns = [
        (r'\b(experiment[s]?|experimental)\b', 'experimental'),
        (r'\b(study|studies)\b', 'study'),
        (r'\b(analysis|analyses)\b', 'analysis'),
        (r'\b(evaluation|assessment)\b', 'evaluation'),
        (r'\b(validation|verification)\b', 'validation'),
        (r'\b(testing|tests?)\b', 'testing'),
        (r'\b(simulation[s]?|modeling)\b', 'simulation'),
        (r'\b(survey[s]?|questionnaire[s]?)\b', 'survey'),
        (r'\b(interview[s]?|observation[s]?)\b', 'qualitative')
    ]
    
    method_count = 0
    method_types = []
    for pattern, method_type in method_patterns:
        matches = re.findall(pattern, text.lower())
        method_count += len(matches)
        if matches:
            method_types.append(method_type)
    
    details['method_appropriateness'] = method_count
    details['method_details'] = list(set(method_types))  # Unique method types
    
    if method_count < 3:
        score -= 0.3
    elif method_count > 6:
        score += 0.1
    
    # Enhanced statistical analysis
    statistical_patterns = [
        (r'\b(statistical|statistically)\b', 'general'),
        (r'\b(significance|significant)\b', 'significance'),
        (r'\b(p.?value[s]?|p\s*[<>=])\b', 'p_value'),
        (r'\b(confidence interval[s]?)\b', 'confidence'),
        (r'\b(correlation[s]?|correlate[s]?)\b', 'correlation'),
        (r'\b(regression[s]?|regress)\b', 'regression'),
        (r'\b(anova|t.?test|chi.?square)\b', 'tests'),
        (r'\b(mean|median|mode|standard deviation)\b', 'descriptive'),
        (r'\b(variance|covariance)\b', 'variance')
    ]
    
    statistical_count = 0
    statistical_methods = []
    for pattern, stat_type in statistical_patterns:
        matches = re.findall(pattern, text.lower())
        statistical_count += len(matches)
        if matches:
            statistical_methods.append(stat_type)
    
    details['statistical_analysis'] = statistical_count
    details['statistical_methods'] = list(set(statistical_methods))
    
    if statistical_count > 0:
        score += 0.1
    if statistical_count > 5:
        score += 0.1
    
    # Enhanced experimental design analysis
    design_patterns = [
        (r'\b(experimental design|study design|research design)\b', 'design'),
        (r'\b(control group[s]?|treatment group[s]?)\b', 'groups'),
        (r'\b(randomized|randomization|random assignment)\b', 'randomization'),
        (r'\b(blinded|blind|double.?blind)\b', 'blinding'),
        (r'\b(placebo|placebo.?controlled)\b', 'placebo'),
        (r'\b(crossover|within.?subject|between.?subject)\b', 'design_type'),
        (r'\b(sample size|power analysis)\b', 'power'),
        (r'\b(inclusion criteria|exclusion criteria)\b', 'criteria')
    ]
    
    design_count = 0
    design_elements = []
    for pattern, design_type in design_patterns:
        matches = re.findall(pattern, text.lower())
        design_count += len(matches)
        if matches:
            design_elements.append(design_type)
    
    details['experimental_design'] = design_count
    details['design_elements'] = list(set(design_elements))
    
    if design_count > 0:
        score += 0.1
    if design_count > 3:
        score += 0.1
    
    # Enhanced limitations analysis
    limitation_patterns = [
        r'\b(limitation[s]?|constraint[s]?)\b',
        r'\b(drawback[s]?|weakness[es]?|shortcoming[s]?)\b',
        r'\b(future work|future research|future studies)\b',
        r'\b(limitation[s]? of this study)\b',
        r'\b(acknowledge[s]?.*limitation[s]?)\b'
    ]
    
    limitation_count = 0
    for pattern in limitation_patterns:
        matches = re.findall(pattern, text.lower())
        limitation_count += len(matches)
    
    details['limitations_acknowledged'] = limitation_count
    
    if limitation_count > 0:
        score += 0.1  # Acknowledging limitations is good practice
    
    return min(1.0, max(0.0, score)), details


def _check_results_significance_enhanced(text: str, paper) -> Tuple[float, Dict]:
    """Enhanced validation of the importance and implications of the findings."""
    score = 1.0
    details = {
        'quantitative_results': 0,
        'significance_statements': 0,
        'comparisons': 0,
        'effect_sizes': 0,
        'confidence_intervals': 0,
        'p_values': 0,
        'specific_results': [],
        'comparison_baselines': [],
        'significance_levels': []
    }
    
    # Find results section content
    results_section = None
    if 'results' in text.lower():
        results_match = re.search(r'results\s*[:\-]?\s*(.*?)(?=discussion|conclusion|$)', text, re.DOTALL | re.IGNORECASE)
        if results_match:
            results_section = results_match.group(1).lower()
    
    if not results_section:
        return 0.4, {'error': 'No clear results section found'}
    
    # Enhanced quantitative results analysis
    quantitative_patterns = [
        (r'\d+\.?\d*%', 'percentage'),
        (r'\d+\.?\d*\s*(increase|decrease|improvement|reduction)', 'change'),
        (r'\d+\.?\d*\s*(times|fold)', 'multiples'),
        (r'significantly\s+(higher|lower|better|worse)', 'significance'),
        (r'p\s*[<>=]\s*0\.\d+', 'p_value'),
        (r'confidence\s+interval', 'confidence'),
        (r'\d+\.?\d*\s*±\s*\d+\.?\d*', 'error_bars'),
        (r'r\s*[=]\s*[+-]?\d+\.?\d*', 'correlation'),
        (r'\d+\.?\d*\s*(mg|ml|g|kg|cm|mm)', 'measurements')
    ]
    
    quantitative_count = 0
    specific_results = []
    for pattern, result_type in quantitative_patterns:
        matches = re.findall(pattern, results_section)
        quantitative_count += len(matches)
        if matches:
            specific_results.extend(matches[:2])  # Store first 2 examples
    
    details['quantitative_results'] = quantitative_count
    details['specific_results'] = specific_results
    
    if quantitative_count < 3:
        score -= 0.3
    elif quantitative_count > 8:
        score += 0.1
    
    # Enhanced significance statements analysis
    significance_patterns = [
        r'\b(significant|significantly)\b',
        r'\b(statistically significant)\b',
        r'\b(important|meaningful|substantial|considerable)\b',
        r'\b(noteworthy|remarkable|striking|dramatic)\b',
        r'\b(highly significant|very significant)\b',
        r'\b(marginally significant|trend toward significance)\b'
    ]
    
    significance_count = 0
    significance_levels = []
    for pattern in significance_patterns:
        matches = re.findall(pattern, results_section)
        significance_count += len(matches)
        significance_levels.extend(matches)
    
    details['significance_statements'] = significance_count
    details['significance_levels'] = significance_levels[:5]  # Store first 5 examples
    
    if significance_count < 2:
        score -= 0.2
    elif significance_count > 5:
        score += 0.1
    
    # Enhanced comparison analysis
    comparison_patterns = [
        r'\b(compared to|versus|vs\.?|in comparison)\b',
        r'\b(baseline|benchmark|state.?of.?the.?art)\b',
        r'\b(previous work|existing methods|prior studies)\b',
        r'\b(control group|treatment group)\b',
        r'\b(superior to|better than|outperforms?)\b',
        r'\b(no significant difference|no difference)\b'
    ]
    
    comparison_count = 0
    comparison_baselines = []
    for pattern in comparison_patterns:
        matches = re.findall(pattern, results_section)
        comparison_count += len(matches)
        comparison_baselines.extend(matches)
    
    details['comparisons'] = comparison_count
    details['comparison_baselines'] = comparison_baselines[:5]  # Store first 5 examples
    
    if comparison_count < 1:
        score -= 0.3
    elif comparison_count > 4:
        score += 0.1
    
    # Enhanced effect size analysis
    effect_size_patterns = [
        r'\b(effect size|cohen.?s d|eta squared)\b',
        r'\b(large effect|medium effect|small effect)\b',
        r'\b(clinical significance|practical significance)\b',
        r'\b(meaningful difference|clinically meaningful)\b'
    ]
    
    effect_size_count = 0
    for pattern in effect_size_patterns:
        matches = re.findall(pattern, results_section)
        effect_size_count += len(matches)
    
    details['effect_sizes'] = effect_size_count
    
    if effect_size_count > 0:
        score += 0.1
    
    # Enhanced confidence interval analysis
    ci_patterns = [
        r'\b(confidence interval[s]?|ci[s]?)\b',
        r'\b(95% confidence|99% confidence)\b',
        r'\b(confidence level[s]?)\b'
    ]
    
    ci_count = 0
    for pattern in ci_patterns:
        matches = re.findall(pattern, results_section)
        ci_count += len(matches)
    
    details['confidence_intervals'] = ci_count
    
    if ci_count > 0:
        score += 0.1
    
    # Enhanced p-value analysis
    p_value_patterns = [
        r'p\s*[<>=]\s*0\.\d+',
        r'p\s*[<>=]\s*0\.0\d+',
        r'p\s*[<>=]\s*0\.00\d+',
        r'\b(p.?value[s]?|p.?val[s]?)\b'
    ]
    
    p_value_count = 0
    for pattern in p_value_patterns:
        matches = re.findall(pattern, results_section)
        p_value_count += len(matches)
    
    details['p_values'] = p_value_count
    
    if p_value_count > 0:
        score += 0.1
    
    # Ensure score doesn't exceed 1.0
    score = min(1.0, score)
    
    return max(0.0, score), details


def _check_theoretical_contribution_enhanced(text: str, paper) -> Tuple[float, Dict]:
    """Enhanced check for theoretical contributions and implications."""
    score = 1.0
    details = {
        'theory_indicators': 0,
        'conceptual_framework': 0,
        'model_development': 0,
        'theoretical_implications': 0,
        'framework_elements': [],
        'theoretical_models': [],
        'conceptual_contributions': []
    }
    
    # Enhanced theoretical framework analysis
    theory_patterns = [
        (r'\b(theoretical|theory|theories)\b', 'general'),
        (r'\b(framework[s]?|model[s]?|concept[s]?)\b', 'framework'),
        (r'\b(principle[s]?|assumption[s]?|postulate[s]?)\b', 'principles'),
        (r'\b(hypothesis|hypotheses)\b', 'hypothesis'),
        (r'\b(theoretical framework|conceptual framework)\b', 'framework_type'),
        (r'\b(paradigm[s]?|paradigmatic)\b', 'paradigm')
    ]
    
    theory_count = 0
    framework_elements = []
    for pattern, theory_type in theory_patterns:
        matches = re.findall(pattern, text.lower())
        theory_count += len(matches)
        if matches:
            framework_elements.append(theory_type)
    
    details['theory_indicators'] = theory_count
    details['framework_elements'] = list(set(framework_elements))
    
    if theory_count < 3:
        score -= 0.3
    elif theory_count > 8:
        score += 0.1
    
    # Enhanced conceptual contributions analysis
    conceptual_patterns = [
        r'\b(conceptual|conceptually|in theory)\b',
        r'\b(theoretically|theoretical basis)\b',
        r'\b(conceptual framework|theoretical framework)\b',
        r'\b(conceptual model|theoretical model)\b',
        r'\b(conceptual contribution[s]?)\b'
    ]
    
    conceptual_count = 0
    conceptual_contributions = []
    for pattern in conceptual_patterns:
        matches = re.findall(pattern, text.lower())
        conceptual_count += len(matches)
        conceptual_contributions.extend(matches)
    
    details['conceptual_framework'] = conceptual_count
    details['conceptual_contributions'] = conceptual_contributions[:3]  # Store first 3 examples
    
    if conceptual_count > 0:
        score += 0.1
    
    # Enhanced model development analysis
    development_patterns = [
        r'\b(propose[s]? a model|develop[s]? a framework)\b',
        r'\b(introduce[s]? a concept|introduce[s]? a model)\b',
        r'\b(theoretical model|conceptual model)\b',
        r'\b(analytical framework|theoretical framework)\b',
        r'\b(new model|novel framework)\b',
        r'\b(model development|framework development)\b'
    ]
    
    development_count = 0
    theoretical_models = []
    for pattern in development_patterns:
        matches = re.findall(pattern, text.lower())
        development_count += len(matches)
        theoretical_models.extend(matches)
    
    details['model_development'] = development_count
    details['theoretical_models'] = theoretical_models[:3]  # Store first 3 examples
    
    if development_count > 0:
        score += 0.2
    elif development_count > 2:
        score += 0.1
    
    # Enhanced theoretical implications analysis
    implication_patterns = [
        r'\b(implications for theory|theoretical implications)\b',
        r'\b(contributes to theory|advances theory)\b',
        r'\b(theoretical significance|theoretical importance)\b',
        r'\b(theoretical contribution[s]?)\b',
        r'\b(theory building|theory development)\b',
        r'\b(extends theory|builds on theory)\b'
    ]
    
    implication_count = 0
    for pattern in implication_patterns:
        matches = re.findall(pattern, text.lower())
        implication_count += len(matches)
    
    details['theoretical_implications'] = implication_count
    
    if implication_count > 0:
        score += 0.1
    
    return min(1.0, max(0.0, score)), details


def _check_practical_implications_enhanced(text: str, paper) -> Tuple[float, Dict]:
    """Enhanced check for practical applications and implications."""
    score = 1.0
    details = {
        'practical_indicators': 0,
        'future_applications': 0,
        'impact_statements': 0,
        'limitations_acknowledged': 0,
        'practical_applications': [],
        'future_work_mentioned': [],
        'impact_areas': []
    }
    
    # Enhanced practical applications analysis
    practical_patterns = [
        (r'\b(practical|practically)\b', 'general'),
        (r'\b(application[s]?|applied|apply)\b', 'application'),
        (r'\b(implement|implementation|deploy|deployment)\b', 'implementation'),
        (r'\b(use case[s]?|use cases)\b', 'use_case'),
        (r'\b(real.?world|real world)\b', 'real_world'),
        (r'\b(industry|industrial|commercial)\b', 'industry'),
        (r'\b(clinical|clinical practice|clinical application)\b', 'clinical'),
        (r'\b(translational|translation)\b', 'translational')
    ]
    
    practical_count = 0
    practical_applications = []
    for pattern, app_type in practical_patterns:
        matches = re.findall(pattern, text.lower())
        practical_count += len(matches)
        if matches:
            practical_applications.append(app_type)
    
    details['practical_indicators'] = practical_count
    details['practical_applications'] = list(set(practical_applications))
    
    if practical_count < 2:
        score -= 0.3
    elif practical_count > 5:
        score += 0.1
    
    # Enhanced future work analysis
    future_patterns = [
        r'\b(future work|future research|future studies)\b',
        r'\b(future applications|potential applications)\b',
        r'\b(possible applications|could be applied)\b',
        r'\b(can be used|useful for)\b',
        r'\b(next steps|further research)\b',
        r'\b(additional work|improvements)\b'
    ]
    
    future_count = 0
    future_work_mentioned = []
    for pattern in future_patterns:
        matches = re.findall(pattern, text.lower())
        future_count += len(matches)
        future_work_mentioned.extend(matches)
    
    details['future_applications'] = future_count
    details['future_work_mentioned'] = future_work_mentioned[:3]  # Store first 3 examples
    
    if future_count > 0:
        score += 0.1
    
    # Enhanced impact analysis
    impact_patterns = [
        (r'\b(impact|implications|significance|importance)\b', 'general'),
        (r'\b(benefit[s]?|beneficial)\b', 'benefit'),
        (r'\b(contribution to society|broader impact)\b', 'societal'),
        (r'\b(clinical impact|therapeutic impact)\b', 'clinical'),
        (r'\b(economic impact|cost.?effectiveness)\b', 'economic'),
        (r'\b(social impact|social implications)\b', 'social')
    ]
    
    impact_count = 0
    impact_areas = []
    for pattern, impact_type in impact_patterns:
        matches = re.findall(pattern, text.lower())
        impact_count += len(matches)
        if matches:
            impact_areas.append(impact_type)
    
    details['impact_statements'] = impact_count
    details['impact_areas'] = list(set(impact_areas))
    
    if impact_count < 2:
        score -= 0.2
    elif impact_count > 4:
        score += 0.1
    
    # Enhanced limitations analysis
    limitation_patterns = [
        r'\b(limitation[s]?|future work|future research)\b',
        r'\b(next steps|further research|additional work)\b',
        r'\b(improvements|enhancements)\b',
        r'\b(constraint[s]?|challenge[s]?)\b'
    ]
    
    limitation_count = 0
    for pattern in limitation_patterns:
        matches = re.findall(pattern, text.lower())
        limitation_count += len(matches)
    
    details['limitations_acknowledged'] = limitation_count
    
    if limitation_count > 0:
        score += 0.1  # Good practice to acknowledge limitations
    
    return min(1.0, max(0.0, score)), details


def _check_research_impact(text: str, paper) -> Tuple[float, Dict]:
    """Check for research impact and broader implications."""
    score = 1.0
    details = {
        'broader_impact': 0,
        'societal_relevance': 0,
        'clinical_relevance': 0,
        'economic_impact': 0,
        'policy_implications': 0,
        'impact_statements': [],
        'societal_benefits': [],
        'clinical_applications': []
    }
    
    # Broader impact analysis
    impact_patterns = [
        (r'\b(broader impact|wider implications)\b', 'broader'),
        (r'\b(societal|society|social)\b', 'societal'),
        (r'\b(clinical|medical|healthcare)\b', 'clinical'),
        (r'\b(economic|cost.?benefit|cost.?effectiveness)\b', 'economic'),
        (r'\b(policy|policymaking|regulatory)\b', 'policy'),
        (r'\b(public health|population health)\b', 'public_health')
    ]
    
    impact_count = 0
    impact_statements = []
    for pattern, impact_type in impact_patterns:
        matches = re.findall(pattern, text.lower())
        impact_count += len(matches)
        if matches:
            impact_statements.append(impact_type)
    
    details['broader_impact'] = impact_count
    details['impact_statements'] = list(set(impact_statements))
    
    if impact_count < 2:
        score -= 0.3
    elif impact_count > 5:
        score += 0.1
    
    # Societal relevance analysis
    societal_patterns = [
        r'\b(societal benefit[s]?|social benefit[s]?)\b',
        r'\b(contribution to society|social impact)\b',
        r'\b(public good|common good)\b',
        r'\b(social justice|equity|accessibility)\b'
    ]
    
    societal_count = 0
    societal_benefits = []
    for pattern in societal_patterns:
        matches = re.findall(pattern, text.lower())
        societal_count += len(matches)
        societal_benefits.extend(matches)
    
    details['societal_relevance'] = societal_count
    details['societal_benefits'] = societal_benefits[:3]  # Store first 3 examples
    
    if societal_count > 0:
        score += 0.1
    
    # Clinical relevance analysis
    clinical_patterns = [
        r'\b(clinical application[s]?|clinical practice)\b',
        r'\b(therapeutic|treatment|therapy)\b',
        r'\b(diagnostic|diagnosis|screening)\b',
        r'\b(patient care|patient outcomes)\b'
    ]
    
    clinical_count = 0
    clinical_applications = []
    for pattern in clinical_patterns:
        matches = re.findall(pattern, text.lower())
        clinical_count += len(matches)
        clinical_applications.extend(matches)
    
    details['clinical_relevance'] = clinical_count
    details['clinical_applications'] = clinical_applications[:3]  # Store first 3 examples
    
    if clinical_count > 0:
        score += 0.1
    
    return min(1.0, max(0.0, score)), details


def _check_reproducibility_quality(text: str, paper) -> Tuple[float, Dict]:
    """Check for reproducibility and transparency quality."""
    score = 1.0
    details = {
        'data_availability': 0,
        'code_availability': 0,
        'method_details': 0,
        'replication_mentions': 0,
        'open_science': 0,
        'data_sources': [],
        'code_repositories': [],
        'replication_elements': []
    }
    
    # Data availability analysis
    data_patterns = [
        (r'\b(data available|data availability)\b', 'general'),
        (r'\b(supplementary data|additional data)\b', 'supplementary'),
        (r'\b(raw data|dataset[s]?)\b', 'raw_data'),
        (r'\b(data repository|data archive)\b', 'repository'),
        (r'\b(open data|public data)\b', 'open_data'),
        (r'\b(data sharing|data sharing policy)\b', 'sharing')
    ]
    
    data_count = 0
    data_sources = []
    for pattern, data_type in data_patterns:
        matches = re.findall(pattern, text.lower())
        data_count += len(matches)
        if matches:
            data_sources.append(data_type)
    
    details['data_availability'] = data_count
    details['data_sources'] = list(set(data_sources))
    
    if data_count > 0:
        score += 0.2
    elif data_count == 0:
        score -= 0.2
    
    # Code availability analysis
    code_patterns = [
        (r'\b(code available|source code)\b', 'general'),
        (r'\b(software|algorithm[s]?)\b', 'software'),
        (r'\b(github|git repository|repository)\b', 'repository'),
        (r'\b(open source|freely available)\b', 'open_source'),
        (r'\b(implementation|implemented)\b', 'implementation')
    ]
    
    code_count = 0
    code_repositories = []
    for pattern, code_type in code_patterns:
        matches = re.findall(pattern, text.lower())
        code_count += len(matches)
        if matches:
            code_repositories.append(code_type)
    
    details['code_availability'] = code_count
    details['code_repositories'] = list(set(code_repositories))
    
    if code_count > 0:
        score += 0.2
    elif code_count == 0:
        score -= 0.1
    
    # Method details analysis
    method_patterns = [
        r'\b(detailed method[s]?|methodology details)\b',
        r'\b(step.?by.?step|detailed procedure[s]?)\b',
        r'\b(reproducible|replicable)\b',
        r'\b(transparent|transparency)\b'
    ]
    
    method_count = 0
    for pattern in method_patterns:
        matches = re.findall(pattern, text.lower())
        method_count += len(matches)
    
    details['method_details'] = method_count
    
    if method_count > 0:
        score += 0.1
    
    # Replication mentions analysis
    replication_patterns = [
        r'\b(replication|replicability|reproducibility)\b',
        r'\b(reproduce|reproduced|reproducing)\b',
        r'\b(replicate|replicated|replicating)\b',
        r'\b(validation|validated|validating)\b'
    ]
    
    replication_count = 0
    replication_elements = []
    for pattern in replication_patterns:
        matches = re.findall(pattern, text.lower())
        replication_count += len(matches)
        replication_elements.extend(matches)
    
    details['replication_mentions'] = replication_count
    details['replication_elements'] = replication_elements[:3]  # Store first 3 examples
    
    if replication_count > 0:
        score += 0.1
    
    # Open science analysis
    open_science_patterns = [
        r'\b(open science|open research)\b',
        r'\b(preprint[s]?|pre.?print[s]?)\b',
        r'\b(peer review|reviewed)\b',
        r'\b(transparent|transparency)\b'
    ]
    
    open_science_count = 0
    for pattern in open_science_patterns:
        matches = re.findall(pattern, text.lower())
        open_science_count += len(matches)
    
    details['open_science'] = open_science_count
    
    if open_science_count > 0:
        score += 0.1
    
    return min(1.0, max(0.0, score)), details


def _generate_quality_feedback_enhanced(novelty_score: float, novelty_details: Dict,
                                       rigor_score: float, rigor_details: Dict,
                                       significance_score: float, significance_details: Dict,
                                       theory_score: float, theory_details: Dict,
                                       practical_score: float, practical_details: Dict,
                                       impact_score: float, impact_details: Dict,
                                       reproducibility_score: float, reproducibility_details: Dict) -> str:
    """Generate detailed feedback for scientific quality with enhanced analysis."""
    feedback_parts = []
    
    # Novelty and originality feedback
    if novelty_score < 0.7:
        feedback_parts.append("NOVELTY: Needs improvement. Consider clearly stating your contributions and how they advance the field.")
        if novelty_details.get('contribution_statements', 0) == 0:
            feedback_parts.append("   - Missing explicit contribution statements")
        if novelty_details.get('problem_statements', 0) < 3:
            feedback_parts.append("   - Insufficient problem identification")
    elif novelty_score >= 0.8:
        feedback_parts.append("NOVELTY: Strong contribution with clear novelty indicators")
    
    # Methodological rigor feedback
    if rigor_score < 0.7:
        feedback_parts.append("RIGOR: Methodological rigor could be enhanced.")
        if rigor_details.get('objectives_clarity', 0) == 0:
            feedback_parts.append("   - Missing clear objectives")
        if rigor_details.get('hypothesis_presence', 0) == 0:
            feedback_parts.append("   - No explicit hypotheses")
        if rigor_details.get('statistical_analysis', 0) == 0:
            feedback_parts.append("   - Limited statistical analysis")
    elif rigor_score >= 0.8:
        feedback_parts.append("RIGOR: Strong methodological foundation")
    
    # Results significance feedback
    if significance_score < 0.7:
        feedback_parts.append("SIGNIFICANCE: Results significance needs better demonstration.")
        if significance_details.get('quantitative_results', 0) < 3:
            feedback_parts.append("   - Insufficient quantitative results")
        if significance_details.get('comparisons', 0) < 1:
            feedback_parts.append("   - Missing comparisons with baselines")
    elif significance_score >= 0.8:
        feedback_parts.append("SIGNIFICANCE: Strong quantitative evidence")
    
    # Theoretical contribution feedback
    if theory_score < 0.7:
        feedback_parts.append("THEORY: Theoretical contribution could be stronger.")
        if theory_details.get('theory_indicators', 0) < 3:
            feedback_parts.append("   - Limited theoretical framework")
        if theory_details.get('model_development', 0) == 0:
            feedback_parts.append("   - No clear model development")
    elif theory_score >= 0.8:
        feedback_parts.append("THEORY: Strong theoretical foundation")
    
    # Practical implications feedback
    if practical_score < 0.7:
        feedback_parts.append("PRACTICAL: Practical implications need better articulation.")
        if practical_details.get('practical_indicators', 0) < 2:
            feedback_parts.append("   - Limited practical applications mentioned")
        if practical_details.get('impact_statements', 0) < 2:
            feedback_parts.append("   - Insufficient impact discussion")
    elif practical_score >= 0.8:
        feedback_parts.append("PRACTICAL: Clear practical applications")
    
    # Research impact feedback
    if impact_score < 0.7:
        feedback_parts.append("IMPACT: Broader impact needs better articulation.")
        if impact_details.get('broader_impact', 0) < 2:
            feedback_parts.append("   - Limited broader impact discussion")
    elif impact_score >= 0.8:
        feedback_parts.append("IMPACT: Strong broader impact")
    
    # Reproducibility feedback
    if reproducibility_score < 0.7:
        feedback_parts.append("REPRODUCIBILITY: Transparency could be improved.")
        if reproducibility_details.get('data_availability', 0) == 0:
            feedback_parts.append("   - No data availability mentioned")
        if reproducibility_details.get('code_availability', 0) == 0:
            feedback_parts.append("   - No code availability mentioned")
    elif reproducibility_score >= 0.8:
        feedback_parts.append("REPRODUCIBILITY: Good transparency practices")
    
    # Overall assessment
    if not any(score < 0.7 for score in [novelty_score, rigor_score, significance_score, theory_score, practical_score]):
        feedback_parts.append("\nOVERALL: Excellent scientific quality with strong contributions across all dimensions.")
    elif all(score >= 0.8 for score in [novelty_score, rigor_score, significance_score]):
        feedback_parts.append("\nOVERALL: High-quality research with particularly strong core contributions.")
    
    return "\n  ".join(feedback_parts)


