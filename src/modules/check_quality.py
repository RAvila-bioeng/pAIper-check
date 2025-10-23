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

# Optional GPT integration
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False

if GPT_AVAILABLE:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
    
    # Add detailed breakdown
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
    
    # GPT enhancement if requested and available
    if use_gpt and GPT_AVAILABLE:
        gpt_analysis_data = {
            'paper_info': {
                'title': paper.title,
                'abstract': paper.abstract,
                'word_count': len(text.split())
            },
            'quality_metrics': basic_result['score_breakdown'],
            'detailed_analysis': basic_result['detailed_analysis']
        }
        basic_result['gpt_analysis_data'] = gpt_analysis_data
        return enhance_quality_with_gpt(paper, basic_result)
    
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
    
    return " ".join(feedback_parts)


# GPT Integration for Enhanced Analysis
if GPT_AVAILABLE:
    class GPTQualityAnalyzer:
        """GPT-4o-mini integration for deep scientific quality analysis."""
        
        def __init__(self):
            self.total_cost = 0.0
            self.total_papers = 0
        
        def should_use_gpt_analysis(self, basic_score: float, feedback: str) -> bool:
            """Decide if GPT analysis is needed based on basic analysis results."""
            if basic_score < 0.7:
                return True
            
            critical_keywords = [
                'needs improvement', 'could be enhanced', 'missing',
                'insufficient', 'limited', 'weak'
            ]
            
            feedback_lower = feedback.lower()
            if any(keyword in feedback_lower for keyword in critical_keywords):
                return True
            
            return False
        
        def analyze_quality(self, paper, gpt_analysis_data: Dict, basic_score: float) -> Dict:
            """Perform deep scientific quality analysis using GPT-4o-mini."""
            
            modules = {
                "novelty_assessment": self._analyze_novelty,
                "rigor_evaluation": self._analyze_rigor,
                "significance_analysis": self._analyze_significance,
                "theoretical_contribution": self._analyze_theory,
                "practical_impact": self._analyze_practical
            }
            
            sub_results = {}
            total_cost = 0
            total_tokens = {'input': 0, 'output': 0}
            
            for name, module_func in modules.items():
                result = module_func(paper, gpt_analysis_data)
                sub_results[name] = result
                if result.get('success'):
                    total_cost += result.get('cost_info', {}).get('cost_usd', 0)
                    total_tokens['input'] += result.get('cost_info', {}).get('input_tokens', 0)
                    total_tokens['output'] += result.get('cost_info', {}).get('output_tokens', 0)
            
            # Aggregate results
            final_score, issues, suggestions, strengths = self._aggregate_results(sub_results, basic_score)
            
            final_analysis = {
                "overall_score": final_score,
                "sub_modules": {name: res.get('analysis', {}) for name, res in sub_results.items()},
                "issues": issues,
                "suggestions": suggestions,
                "strengths": strengths,
                "final_verdict": self._get_final_verdict(final_score)
            }
            
            return {
                'success': True,
                'analysis': final_analysis,
                'cost_info': {
                    'cost_usd': round(total_cost, 4),
                    'input_tokens': total_tokens['input'],
                    'output_tokens': total_tokens['output'],
                    'total_tokens': total_tokens['input'] + total_tokens['output']
                },
                'model': 'gpt-4o-mini'
            }
        
        def _analyze_novelty(self, paper, gpt_analysis_data: Dict) -> Dict:
            """Analyze novelty and originality using GPT."""
            system_prompt = "You are an expert scientific reviewer analyzing novelty and originality. Respond in JSON."
            
            prompt = (
                f"Analyze the novelty and originality of this research paper.\n"
                f"**Title:** {paper.title}\n"
                f"**Abstract:** {paper.abstract}\n\n"
                "YOUR TASK:\n"
                "1. Rate novelty from 0.0 to 1.0.\n"
                "2. Identify specific novel contributions.\n"
                "3. Assess how the work advances the field.\n"
                "4. Note any gaps in novelty claims.\n"
                'RESPOND IN JSON FORMAT:\n'
                '{"score": <float>, "novel_contributions": ["<contribution1>", "<contribution2>"], "field_advancement": "<string>", "gaps": ["<gap1>", "<gap2>"]}'
            )
            return self._call_gpt(prompt, system_prompt)
        
        def _analyze_rigor(self, paper, gpt_analysis_data: Dict) -> Dict:
            """Analyze methodological rigor using GPT."""
            system_prompt = "You are an expert scientific reviewer analyzing methodological rigor. Respond in JSON."
            
            methodology_content = paper.get_section_content('methodology') or "Methodology section not found"
            
            prompt = (
                f"Analyze the methodological rigor of this research.\n"
                f"**Title:** {paper.title}\n"
                f"**Methodology:** {methodology_content[:1000]}...\n\n"
                "YOUR TASK:\n"
                "1. Rate methodological rigor from 0.0 to 1.0.\n"
                "2. Assess experimental design quality.\n"
                "3. Evaluate statistical analysis appropriateness.\n"
                "4. Identify methodological limitations.\n"
                'RESPOND IN JSON FORMAT:\n'
                '{"score": <float>, "design_quality": "<string>", "statistical_appropriateness": "<string>", "limitations": ["<limitation1>", "<limitation2>"]}'
            )
            return self._call_gpt(prompt, system_prompt)
        
        def _analyze_significance(self, paper, gpt_analysis_data: Dict) -> Dict:
            """Analyze results significance using GPT."""
            system_prompt = "You are an expert scientific reviewer analyzing results significance. Respond in JSON."
            
            results_content = paper.get_section_content('results') or "Results section not found"
            
            prompt = (
                f"Analyze the significance of the research results.\n"
                f"**Title:** {paper.title}\n"
                f"**Results:** {results_content[:1000]}...\n\n"
                "YOUR TASK:\n"
                "1. Rate results significance from 0.0 to 1.0.\n"
                "2. Assess quantitative evidence strength.\n"
                "3. Evaluate comparison with baselines.\n"
                "4. Identify significance limitations.\n"
                'RESPOND IN JSON FORMAT:\n'
                '{"score": <float>, "evidence_strength": "<string>", "baseline_comparisons": "<string>", "limitations": ["<limitation1>", "<limitation2>"]}'
            )
            return self._call_gpt(prompt, system_prompt)
        
        def _analyze_theory(self, paper, gpt_analysis_data: Dict) -> Dict:
            """Analyze theoretical contribution using GPT."""
            system_prompt = "You are an expert scientific reviewer analyzing theoretical contributions. Respond in JSON."
            
            prompt = (
                f"Analyze the theoretical contribution of this research.\n"
                f"**Title:** {paper.title}\n"
                f"**Abstract:** {paper.abstract}\n\n"
                "YOUR TASK:\n"
                "1. Rate theoretical contribution from 0.0 to 1.0.\n"
                "2. Identify theoretical frameworks used.\n"
                "3. Assess theoretical implications.\n"
                "4. Note theoretical limitations.\n"
                'RESPOND IN JSON FORMAT:\n'
                '{"score": <float>, "frameworks": ["<framework1>", "<framework2>"], "implications": "<string>", "limitations": ["<limitation1>", "<limitation2>"]}'
            )
            return self._call_gpt(prompt, system_prompt)
        
        def _analyze_practical(self, paper, gpt_analysis_data: Dict) -> Dict:
            """Analyze practical implications using GPT."""
            system_prompt = "You are an expert scientific reviewer analyzing practical implications. Respond in JSON."
            
            prompt = (
                f"Analyze the practical implications of this research.\n"
                f"**Title:** {paper.title}\n"
                f"**Abstract:** {paper.abstract}\n\n"
                "YOUR TASK:\n"
                "1. Rate practical implications from 0.0 to 1.0.\n"
                "2. Identify potential applications.\n"
                "3. Assess broader impact.\n"
                "4. Note practical limitations.\n"
                'RESPOND IN JSON FORMAT:\n'
                '{"score": <float>, "applications": ["<app1>", "<app2>"], "broader_impact": "<string>", "limitations": ["<limitation1>", "<limitation2>"]}'
            )
            return self._call_gpt(prompt, system_prompt)
        
        def _call_gpt(self, prompt: str, system_prompt: str, max_tokens=400) -> Dict:
            """Make a structured call to the OpenAI API."""
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                )
                
                analysis_text = response.choices[0].message.content
                analysis_json = json.loads(analysis_text)
                
                cost = self._calculate_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                
                return {
                    'success': True,
                    'analysis': analysis_json,
                    'cost_info': {
                        'cost_usd': round(cost, 4),
                        'input_tokens': response.usage.prompt_tokens,
                        'output_tokens': response.usage.completion_tokens,
                    }
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'analysis': {'score': 0.0, 'feedback': f'Analysis failed: {e}'},
                    'cost_info': {'cost_usd': 0.0, 'input_tokens': 0, 'output_tokens': 0}
                }
        
        def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
            """Calculate cost in USD for the API call."""
            COST_PER_1M_INPUT = 0.15
            COST_PER_1M_OUTPUT = 0.60
            input_cost = (input_tokens / 1_000_000) * COST_PER_1M_INPUT
            output_cost = (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT
            return input_cost + output_cost
        
        def _aggregate_results(self, sub_results: Dict, basic_score: float) -> tuple:
            """Aggregate results from all sub-modules."""
            issues, suggestions, strengths = [], [], []
            
            weights = {"novelty_assessment": 0.25, "rigor_evaluation": 0.20, "significance_analysis": 0.20, 
                      "theoretical_contribution": 0.15, "practical_impact": 0.20}
            
            total_score, total_weight = 0, 0
            
            for name, result in sub_results.items():
                if result.get('success'):
                    analysis = result.get('analysis', {})
                    score = analysis.get('score', 0.0)
                    
                    total_score += score * weights.get(name, 0)
                    total_weight += weights.get(name, 0)
                    
                    if score < 0.6:
                        issues.append(f"Weak {name.replace('_', ' ')}: {analysis.get('feedback', 'No feedback')}")
                    elif score >= 0.8:
                        strengths.append(f"Strong {name.replace('_', ' ')}: {analysis.get('feedback', 'No feedback')}")
            
            final_score = (total_score / total_weight) if total_weight > 0 else basic_score
            
            return final_score, issues, suggestions, strengths
        
        def _get_final_verdict(self, score: float) -> str:
            if score >= 0.85: return "EXCELLENT"
            if score >= 0.7: return "GOOD"
            if score >= 0.5: return "FAIR"
            return "POOR"
    
    def enhance_quality_with_gpt(paper, basic_result: Dict, force_analysis: bool = False) -> Dict:
        """Main integration function: Enhance basic quality analysis with GPT-4o-mini."""
        
        analyzer = GPTQualityAnalyzer()
        
        basic_score = basic_result.get('score', 0.0)
        basic_feedback = basic_result.get('feedback', '')
        gpt_analysis_data = basic_result.get('gpt_analysis_data', {})
        
        # Decide if GPT analysis is needed
        needs_gpt = force_analysis or analyzer.should_use_gpt_analysis(basic_score, basic_feedback)
        
        if not needs_gpt:
            basic_result['gpt_analysis'] = {
                'used': False,
                'reason': 'Basic analysis sufficient (score >= 0.7)',
                'cost_saved': 0.003
            }
            return basic_result
        
        # Perform GPT analysis
        gpt_result = analyzer.analyze_quality(paper, gpt_analysis_data, basic_score)
        
        # Enhance feedback
        if gpt_result.get('success'):
            enhanced_feedback = analyzer.format_gpt_feedback(gpt_result, basic_feedback)
            basic_result['feedback'] = enhanced_feedback
            
            # Update score if GPT provides better assessment
            gpt_score = gpt_result.get('analysis', {}).get('overall_score')
            if gpt_score is not None:
                # Weighted average: 40% basic, 60% GPT
                basic_result['score'] = (basic_score * 0.4) + (gpt_score * 0.6)
                basic_result['score_breakdown']['gpt_enhanced'] = True
        
        # Add GPT analysis to result
        basic_result['gpt_analysis'] = gpt_result
        
        return basic_result
    
    def format_gpt_feedback(self, gpt_result: Dict, basic_feedback: str) -> str:
        """Format GPT analysis into readable feedback."""
        if not gpt_result.get('success'):
            return basic_feedback + f" [GPT Analysis Failed: {gpt_result.get('error', 'Unknown error')}]"
        
        analysis = gpt_result.get('analysis', {})
        feedback_parts = [basic_feedback]
        
        # Add GPT score comparison
        gpt_score = analysis.get('overall_score', 0)
        feedback_parts.append(f"\n\n--- DEEP AI ANALYSIS (GPT-4o-mini) ---")
        feedback_parts.append(f"Enhanced Quality Score: {gpt_score:.2f}/1.0")
        
        # Add sub-module scores
        if 'sub_modules' in analysis:
            feedback_parts.append("\nQuality Sub-modules:")
            for name, sub in analysis['sub_modules'].items():
                feedback_parts.append(f"  - {name.replace('_', ' ').title()}: {sub.get('score', 0):.2f}")
        
        # Add critical issues
        issues = analysis.get('issues', [])
        if issues:
            feedback_parts.append(f"\n🔴 CRITICAL ISSUES ({len(issues)}):")
            for issue in issues[:3]:
                feedback_parts.append(f"  • {issue}")
        
        # Add strengths
        strengths = analysis.get('strengths', [])
        if strengths:
            feedback_parts.append(f"\n✓ STRENGTHS: {', '.join(strengths[:3])}")
        
        # Add verdict
        verdict = analysis.get('final_verdict', 'N/A')
        feedback_parts.append(f"\nFinal Verdict: {verdict}")
        
        # Add cost info
        cost_info = gpt_result.get('cost_info', {})
        feedback_parts.append(f"\n[Analysis cost: ${cost_info.get('cost_usd', 0):.4f} | Tokens: {cost_info.get('total_tokens', 0)}]")
        
        return "\n".join(feedback_parts)
