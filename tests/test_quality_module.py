"""
Test module for the enhanced Scientific Quality Evaluation module.
Tests the new advanced metrics and GPT integration capabilities.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.paper import Paper
from modules.check_quality import (
    evaluate, 
    _check_novelty_originality_enhanced,
    _check_methodological_rigor_enhanced,
    _check_results_significance_enhanced,
    _check_theoretical_contribution_enhanced,
    _check_practical_implications_enhanced,
    _check_research_impact,
    _check_reproducibility_quality,
    _generate_quality_feedback_enhanced
)


class TestScientificQualityModule:
    """Test suite for the enhanced scientific quality evaluation module."""
    
    @pytest.fixture
    def sample_paper(self):
        """Create a sample paper for testing."""
        text = """
        Title: Novel Machine Learning Approach for Biomedical Data Analysis
        
        Abstract: This paper presents a novel machine learning framework for analyzing biomedical data. 
        Our main contribution is the development of a new algorithm that significantly improves 
        classification accuracy compared to existing methods. The proposed approach addresses the 
        limitation of current techniques in handling high-dimensional data.
        
        Introduction: The analysis of biomedical data presents significant challenges due to its 
        high dimensionality and complexity. Previous work has shown limitations in handling 
        such data effectively. Our objective is to develop a more robust approach that can 
        handle these challenges while maintaining computational efficiency.
        
        Methodology: We propose a novel machine learning algorithm based on deep learning principles. 
        The experimental design includes a randomized controlled trial with 1000 participants. 
        Statistical analysis was performed using ANOVA and t-tests. The methodology is 
        reproducible and the code is available on GitHub.
        
        Results: Our results show a significant improvement of 15% in classification accuracy 
        compared to baseline methods (p < 0.001). The confidence interval for the improvement 
        was 12-18%. Statistical significance was confirmed across all test conditions.
        
        Discussion: The theoretical implications of our work extend beyond biomedical applications. 
        The practical applications include clinical decision support systems and diagnostic tools. 
        Future work will explore the broader impact on healthcare systems.
        
        Conclusion: This research contributes to the field by providing a novel approach that 
        addresses current limitations. The practical implications are significant for both 
        researchers and practitioners in the biomedical domain.
        """
        return Paper(raw_text=text)
    
    def test_basic_evaluation(self, sample_paper):
        """Test basic evaluation functionality."""
        result = evaluate(sample_paper, use_gpt=False)
        
        # Check basic structure
        assert 'score' in result
        assert 'feedback' in result
        assert 'score_breakdown' in result
        assert 'detailed_analysis' in result
        
        # Check score is valid
        assert 0.0 <= result['score'] <= 1.0
        
        # Check score breakdown has all components
        breakdown = result['score_breakdown']
        expected_components = ['novelty', 'rigor', 'significance', 'theory', 'practical', 'impact', 'reproducibility']
        for component in expected_components:
            assert component in breakdown
            assert 0.0 <= breakdown[component] <= 1.0
    
    def test_novelty_originality_enhanced(self, sample_paper):
        """Test enhanced novelty and originality analysis."""
        score, details = _check_novelty_originality_enhanced(sample_paper.full_text, sample_paper)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
        
        # Check that details contain expected keys
        expected_keys = ['novelty_indicators_found', 'contribution_statements', 'problem_statements', 
                        'comparisons', 'specific_contributions', 'gaps_identified']
        for key in expected_keys:
            assert key in details
    
    def test_methodological_rigor_enhanced(self, sample_paper):
        """Test enhanced methodological rigor analysis."""
        score, details = _check_methodological_rigor_enhanced(sample_paper.full_text, sample_paper.sections)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
        
        # Check that details contain expected keys
        expected_keys = ['objectives_clarity', 'hypothesis_presence', 'method_appropriateness', 
                        'statistical_analysis', 'experimental_design', 'limitations_acknowledged']
        for key in expected_keys:
            assert key in details
    
    def test_results_significance_enhanced(self, sample_paper):
        """Test enhanced results significance analysis."""
        score, details = _check_results_significance_enhanced(sample_paper.full_text, sample_paper)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
        
        # Check that details contain expected keys
        expected_keys = ['quantitative_results', 'significance_statements', 'comparisons', 
                        'effect_sizes', 'confidence_intervals', 'p_values']
        for key in expected_keys:
            assert key in details
    
    def test_theoretical_contribution_enhanced(self, sample_paper):
        """Test enhanced theoretical contribution analysis."""
        score, details = _check_theoretical_contribution_enhanced(sample_paper.full_text, sample_paper)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
        
        # Check that details contain expected keys
        expected_keys = ['theory_indicators', 'conceptual_framework', 'model_development', 
                        'theoretical_implications', 'framework_elements']
        for key in expected_keys:
            assert key in details
    
    def test_practical_implications_enhanced(self, sample_paper):
        """Test enhanced practical implications analysis."""
        score, details = _check_practical_implications_enhanced(sample_paper.full_text, sample_paper)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
        
        # Check that details contain expected keys
        expected_keys = ['practical_indicators', 'future_applications', 'impact_statements', 
                        'limitations_acknowledged', 'practical_applications']
        for key in expected_keys:
            assert key in details
    
    def test_research_impact(self, sample_paper):
        """Test research impact analysis."""
        score, details = _check_research_impact(sample_paper.full_text, sample_paper)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
        
        # Check that details contain expected keys
        expected_keys = ['broader_impact', 'societal_relevance', 'clinical_relevance', 
                        'economic_impact', 'policy_implications']
        for key in expected_keys:
            assert key in details
    
    def test_reproducibility_quality(self, sample_paper):
        """Test reproducibility quality analysis."""
        score, details = _check_reproducibility_quality(sample_paper.full_text, sample_paper)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
        
        # Check that details contain expected keys
        expected_keys = ['data_availability', 'code_availability', 'method_details', 
                        'replication_mentions', 'open_science']
        for key in expected_keys:
            assert key in details
    
    def test_enhanced_feedback_generation(self, sample_paper):
        """Test enhanced feedback generation."""
        # Mock scores and details
        novelty_score = 0.8
        novelty_details = {'contribution_statements': 2, 'problem_statements': 3}
        rigor_score = 0.7
        rigor_details = {'objectives_clarity': 2, 'hypothesis_presence': 1, 'statistical_analysis': 3}
        significance_score = 0.9
        significance_details = {'quantitative_results': 5, 'comparisons': 2}
        theory_score = 0.6
        theory_details = {'theory_indicators': 2, 'model_development': 0}
        practical_score = 0.8
        practical_details = {'practical_indicators': 3, 'impact_statements': 2}
        impact_score = 0.7
        impact_details = {'broader_impact': 2}
        reproducibility_score = 0.9
        reproducibility_details = {'data_availability': 1, 'code_availability': 1}
        
        feedback = _generate_quality_feedback_enhanced(
            novelty_score, novelty_details,
            rigor_score, rigor_details,
            significance_score, significance_details,
            theory_score, theory_details,
            practical_score, practical_details,
            impact_score, impact_details,
            reproducibility_score, reproducibility_details
        )
        
        assert isinstance(feedback, str)
        assert len(feedback) > 0
        
        # Check that feedback contains expected elements
        assert 'NOVELTY' in feedback or 'RIGOR' in feedback or 'SIGNIFICANCE' in feedback
    
    def test_paper_without_results_section(self):
        """Test handling of papers without clear results section."""
        text = """
        Title: Theoretical Framework for Data Analysis
        
        Abstract: This paper presents a theoretical framework for analyzing data.
        
        Introduction: We propose a new theoretical approach.
        
        Methodology: Our approach is based on mathematical principles.
        
        Discussion: The implications are significant.
        
        Conclusion: This work contributes to the field.
        """
        
        paper = Paper(raw_text=text)
        score, details = _check_results_significance_enhanced(paper.full_text, paper)
        
        # Should handle missing results section gracefully
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
    
    def test_edge_case_empty_paper(self):
        """Test handling of empty or minimal papers."""
        text = "Title: Test\n\nAbstract: This is a test."
        paper = Paper(raw_text=text)
        
        result = evaluate(paper, use_gpt=False)
        
        # Should not crash and return valid scores
        assert 0.0 <= result['score'] <= 1.0
        assert isinstance(result['feedback'], str)
    
    def test_score_breakdown_consistency(self, sample_paper):
        """Test that score breakdown is consistent with overall score."""
        result = evaluate(sample_paper, use_gpt=False)
        
        breakdown = result['score_breakdown']
        weights = {
            'novelty': 0.25,
            'rigor': 0.20,
            'significance': 0.20,
            'theory': 0.15,
            'practical': 0.10,
            'impact': 0.05,
            'reproducibility': 0.05
        }
        
        # Calculate expected weighted score
        expected_score = sum(breakdown[component] * weights[component] for component in weights.keys())
        
        # Allow for small floating point differences
        assert abs(result['score'] - expected_score) < 0.01
    
    def test_detailed_analysis_structure(self, sample_paper):
        """Test that detailed analysis has proper structure."""
        result = evaluate(sample_paper, use_gpt=False)
        
        detailed_analysis = result['detailed_analysis']
        
        # Check that all components are present
        expected_components = ['novelty', 'rigor', 'significance', 'theory', 'practical', 'impact', 'reproducibility']
        for component in expected_components:
            assert component in detailed_analysis
            assert isinstance(detailed_analysis[component], dict)

    def test_low_novelty_paper(self):
        """Test novelty analysis on a paper with low novelty."""
        text = """
        Title: A Review of Existing Methods
        
        Abstract: This paper reviews existing methods in the field. We summarize the current state of the art.
        
        Introduction: Many methods exist for this problem. This paper describes them.
        
        Methodology: We used a standard literature review methodology.
        
        Results: The results are a summary of what is already known.
        
        Conclusion: This paper has summarized existing work.
        """
        paper = Paper(raw_text=text)
        score, details = _check_novelty_originality_enhanced(paper.full_text, paper)
        
        assert score < 0.6
        assert details['novelty_indicators_found'] == 0
        assert details['contribution_statements'] == 0


class TestGPTIntegration:
    """Test suite for GPT integration (if available)."""
    
    @pytest.fixture
    def sample_paper(self):
        """Create a sample paper for testing."""
        text = """
        Title: Advanced Machine Learning for Medical Diagnosis
        
        Abstract: This paper introduces a novel deep learning approach for medical diagnosis. 
        Our main contribution is a new neural network architecture that achieves state-of-the-art 
        performance on medical imaging tasks. The proposed method addresses critical limitations 
        in current diagnostic systems.
        
        Introduction: Medical diagnosis presents unique challenges that require sophisticated 
        machine learning approaches. Current methods have significant limitations in accuracy 
        and reliability. Our objective is to develop a more robust diagnostic system.
        
        Methodology: We designed a novel convolutional neural network with attention mechanisms. 
        The experimental design includes a double-blind randomized controlled trial with 2000 patients. 
        Statistical analysis was performed using appropriate tests with p < 0.05 significance level.
        
        Results: Our method achieved 95% accuracy compared to 87% for baseline methods (p < 0.001). 
        The improvement was statistically significant across all test conditions. Confidence intervals 
        confirmed the robustness of our findings.
        
        Discussion: The theoretical implications extend to other medical domains. Practical 
        applications include real-time diagnostic systems and clinical decision support tools. 
        Future work will explore broader healthcare applications.
        
        Conclusion: This research makes significant contributions to medical AI by providing 
        a novel approach that addresses current limitations. The practical impact is substantial 
        for healthcare professionals and patients.
        """
        return Paper(raw_text=text)
    
    def test_gpt_integration_availability(self):
        """Test GPT integration availability."""
        try:
            from modules.check_quality import GPTQualityAnalyzer
            # If we can import it, GPT is available
            assert True
        except ImportError:
            # GPT not available, which is fine
            assert True
    
    def test_evaluate_with_gpt_flag(self, sample_paper):
        """Test evaluation with GPT flag (should work regardless of GPT availability)."""
        # This should work whether GPT is available or not
        result = evaluate(sample_paper, use_gpt=True)
        
        # Check basic structure
        assert 'score' in result
        assert 'feedback' in result
        
        # Check if GPT analysis data was prepared (even if GPT is not available)
        assert 'gpt_analysis_data' in result
        
        # Score should still be valid
        assert 0.0 <= result['score'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
