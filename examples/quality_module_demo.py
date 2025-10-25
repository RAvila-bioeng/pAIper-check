#!/usr/bin/env python3
"""
Demo script for the enhanced Scientific Quality Evaluation Module.
This script demonstrates the new features and capabilities.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.paper import Paper
from modules.check_quality import evaluate


def main():
    """Demonstrate the enhanced scientific quality evaluation module."""
    
    print("Enhanced Scientific Quality Evaluation Module Demo")
    print("=" * 60)
    
    # Sample paper text
    sample_text = """
    Title: Novel Machine Learning Approach for Biomedical Data Analysis
    
    Abstract: This paper presents a novel machine learning framework for analyzing biomedical data. 
    Our main contribution is the development of a new algorithm that significantly improves 
    classification accuracy compared to existing methods. The proposed approach addresses the 
    limitation of current techniques in handling high-dimensional data. This work contributes 
    to the field by providing a robust solution for biomedical data analysis challenges.
    
    Introduction: The analysis of biomedical data presents significant challenges due to its 
    high dimensionality and complexity. Previous work has shown limitations in handling 
    such data effectively. Our objective is to develop a more robust approach that can 
    handle these challenges while maintaining computational efficiency. We hypothesize that 
    our novel approach will outperform existing methods by at least 15%.
    
    Methodology: We propose a novel machine learning algorithm based on deep learning principles. 
    The experimental design includes a randomized controlled trial with 1000 participants. 
    Statistical analysis was performed using ANOVA and t-tests with p < 0.05 significance level. 
    The methodology is reproducible and the code is available on GitHub. We used a double-blind 
    experimental design with control and treatment groups.
    
    Results: Our results show a significant improvement of 15% in classification accuracy 
    compared to baseline methods (p < 0.001, 95% confidence interval: 12-18%). The improvement 
    was statistically significant across all test conditions. We observed a large effect size 
    (Cohen's d = 0.8) indicating practical significance. The results demonstrate superior 
    performance compared to state-of-the-art methods.
    
    Discussion: The theoretical implications of our work extend beyond biomedical applications. 
    The practical applications include clinical decision support systems and diagnostic tools. 
    Future work will explore the broader impact on healthcare systems. The societal benefits 
    include improved patient outcomes and reduced healthcare costs. Clinical applications 
    could revolutionize medical diagnosis.
    
    Conclusion: This research contributes to the field by providing a novel approach that 
    addresses current limitations. The practical implications are significant for both 
    researchers and practitioners in the biomedical domain. The broader impact extends 
    to healthcare policy and clinical practice guidelines.
    """
    
    # Create paper object
    paper = Paper(raw_text=sample_text)
    
    print(f"Analyzing paper: {paper.title}")
    print(f"Word count: {len(paper.full_text.split())}")
    print()
    
    # Basic evaluation
    print("Running basic scientific quality evaluation...")
    result = evaluate(paper, use_gpt=False)
    
    print(f"Overall Score: {result['score']:.2f}/1.0")
    print()
    
    # Score breakdown
    print("Score Breakdown:")
    breakdown = result['score_breakdown']
    for component, score in breakdown.items():
        status = "GOOD" if score >= 0.8 else "FAIR" if score >= 0.6 else "POOR"
        print(f"  {status} {component.title()}: {score:.2f}")
    print()
    
    # Detailed analysis
    print("Detailed Analysis:")
    detailed = result['detailed_analysis']
    
    # Novelty analysis
    novelty = detailed['novelty']
    print(f"  Novelty Indicators Found: {novelty['novelty_indicators_found']}")
    print(f"  Contribution Statements: {novelty['contribution_statements']}")
    print(f"  Problem Statements: {novelty['problem_statements']}")
    print(f"  Comparisons: {novelty['comparisons']}")
    if novelty['specific_contributions']:
        print(f"  Specific Contributions: {novelty['specific_contributions'][:3]}")
    print()
    
    # Rigor analysis
    rigor = detailed['rigor']
    print(f"  Objectives Clarity: {rigor['objectives_clarity']}")
    print(f"  Hypothesis Presence: {rigor['hypothesis_presence']}")
    print(f"  Method Appropriateness: {rigor['method_appropriateness']}")
    print(f"  Statistical Analysis: {rigor['statistical_analysis']}")
    print(f"  Experimental Design: {rigor['experimental_design']}")
    if rigor['method_details']:
        print(f"  Method Types: {rigor['method_details']}")
    if rigor['statistical_methods']:
        print(f"  Statistical Methods: {rigor['statistical_methods']}")
    print()
    
    # Significance analysis
    significance = detailed['significance']
    print(f"  Quantitative Results: {significance['quantitative_results']}")
    print(f"  Significance Statements: {significance['significance_statements']}")
    print(f"  Comparisons: {significance['comparisons']}")
    print(f"  Effect Sizes: {significance['effect_sizes']}")
    print(f"  Confidence Intervals: {significance['confidence_intervals']}")
    print(f"  P-Values: {significance['p_values']}")
    if significance['specific_results']:
        print(f"  Specific Results: {significance['specific_results'][:3]}")
    print()
    
    # Theory analysis
    theory = detailed['theory']
    print(f"  Theory Indicators: {theory['theory_indicators']}")
    print(f"  Conceptual Framework: {theory['conceptual_framework']}")
    print(f"  Model Development: {theory['model_development']}")
    print(f"  Theoretical Implications: {theory['theoretical_implications']}")
    if theory['framework_elements']:
        print(f"  Framework Elements: {theory['framework_elements']}")
    print()
    
    # Practical analysis
    practical = detailed['practical']
    print(f"  Practical Indicators: {practical['practical_indicators']}")
    print(f"  Future Applications: {practical['future_applications']}")
    print(f"  Impact Statements: {practical['impact_statements']}")
    print(f"  Limitations Acknowledged: {practical['limitations_acknowledged']}")
    if practical['practical_applications']:
        print(f"  Application Types: {practical['practical_applications']}")
    print()
    
    # Impact analysis
    impact = detailed['impact']
    print(f"  Broader Impact: {impact['broader_impact']}")
    print(f"  Societal Relevance: {impact['societal_relevance']}")
    print(f"  Clinical Relevance: {impact['clinical_relevance']}")
    print(f"  Economic Impact: {impact['economic_impact']}")
    if impact['impact_statements']:
        print(f"  Impact Types: {impact['impact_statements']}")
    print()
    
    # Reproducibility analysis
    reproducibility = detailed['reproducibility']
    print(f"  Data Availability: {reproducibility['data_availability']}")
    print(f"  Code Availability: {reproducibility['code_availability']}")
    print(f"  Method Details: {reproducibility['method_details']}")
    print(f"  Replication Mentions: {reproducibility['replication_mentions']}")
    print(f"  Open Science: {reproducibility['open_science']}")
    if reproducibility['data_sources']:
        print(f"  Data Sources: {reproducibility['data_sources']}")
    print()
    
    # Enhanced feedback
    print("Enhanced Feedback:")
    print(result['feedback'])
    print()
    
    # GPT analysis data preparation
    print("GPT Analysis Data Prepared:")
    gpt_data = result.get('gpt_analysis_data', {})
    if gpt_data:
        print(f"  Paper Info: {gpt_data.get('paper_info', {}).get('title', 'N/A')}")
        print(f"  Quality Metrics: {len(gpt_data.get('quality_metrics', {}))} components")
        print(f"  Detailed Analysis: {len(gpt_data.get('detailed_analysis', {}))} components")
        print("  Ready for GPT analysis (if OpenAI API key is available)")
    else:
        print("  No GPT analysis data prepared")
    print()
    
    print("Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("  Enhanced evaluation with 7 quality dimensions")
    print("  Detailed analysis with specific metrics")
    print("  Comprehensive score breakdown")
    print("  Enhanced feedback with visual indicators")
    print("  GPT analysis data preparation")
    print("  Advanced pattern recognition")
    print("  Context-aware scoring")


if __name__ == "__main__":
    main()
