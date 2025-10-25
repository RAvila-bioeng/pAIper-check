# Scientific Quality Evaluation Module Documentation

## Overview

The Scientific Quality Evaluation Module (`check_quality.py`) is the core component of pAIper-check, focusing on assessing the intrinsic scientific rigor, novelty, and impact of academic papers. This module has been significantly enhanced with advanced metrics and optional GPT integration for deeper analysis.

## Key Features

### 🎯 **Enhanced Evaluation Metrics**
- **Novelty & Originality**: Advanced pattern recognition for contribution statements and research gaps
- **Methodological Rigor**: Comprehensive analysis of experimental design, statistical methods, and limitations
- **Results Significance**: Quantitative evidence assessment with effect sizes and confidence intervals
- **Theoretical Contribution**: Framework development and theoretical implications analysis
- **Practical Implications**: Real-world applications and broader impact assessment
- **Research Impact**: Societal, clinical, and economic impact evaluation
- **Reproducibility Quality**: Data availability, code transparency, and open science practices

### 🤖 **GPT-4o-mini Integration**
- Optional deep analysis using OpenAI's GPT-4o-mini model
- Cost-optimized implementation (~$0.003-0.005 per paper)
- Intelligent triggering based on basic analysis results
- Modular sub-analysis for each quality dimension

### 📊 **Detailed Analytics**
- Comprehensive score breakdown by component
- Detailed analysis with specific examples and metrics
- Enhanced feedback with actionable recommendations
- Visual indicators (🔴, ✅, 🎉) for quick assessment

## Module Architecture

### Core Functions

#### `evaluate(paper, use_gpt=False)`
Main evaluation function that orchestrates all quality assessments.

**Parameters:**
- `paper`: Paper object with text content
- `use_gpt`: Whether to use GPT for deeper analysis (optional)

**Returns:**
- Dictionary with score, feedback, breakdown, and detailed analysis

#### Enhanced Analysis Functions

1. **`_check_novelty_originality_enhanced(text, paper)`**
   - Analyzes contribution statements and novelty indicators
   - Detects research gaps and problem statements
   - Evaluates comparison with existing work
   - Returns score and detailed metrics

2. **`_check_methodological_rigor_enhanced(text, sections)`**
   - Assesses objectives clarity and hypothesis presence
   - Evaluates experimental design quality
   - Analyzes statistical methods and limitations acknowledgment
   - Returns comprehensive rigor assessment

3. **`_check_results_significance_enhanced(text, paper)`**
   - Quantifies results presentation and significance statements
   - Evaluates baseline comparisons and effect sizes
   - Assesses confidence intervals and p-values
   - Returns detailed significance metrics

4. **`_check_theoretical_contribution_enhanced(text, paper)`**
   - Analyzes theoretical frameworks and model development
   - Evaluates conceptual contributions
   - Assesses theoretical implications
   - Returns theoretical quality metrics

5. **`_check_practical_implications_enhanced(text, paper)`**
   - Evaluates practical applications and future work
   - Assesses broader impact statements
   - Analyzes limitations acknowledgment
   - Returns practical relevance metrics

6. **`_check_research_impact(text, paper)`**
   - Evaluates broader impact and societal relevance
   - Assesses clinical and economic implications
   - Analyzes policy implications
   - Returns impact assessment metrics

7. **`_check_reproducibility_quality(text, paper)`**
   - Evaluates data and code availability
   - Assesses method transparency
   - Analyzes open science practices
   - Returns reproducibility metrics

### GPT Integration (Optional)

#### `GPTQualityAnalyzer` Class
- Manages GPT-4o-mini analysis for deep quality evaluation
- Cost tracking and optimization
- Modular sub-analysis approach
- Intelligent analysis triggering

#### Key Methods:
- `should_use_gpt_analysis()`: Determines if GPT analysis is needed
- `analyze_quality()`: Performs comprehensive GPT analysis
- `_analyze_novelty()`: GPT-powered novelty assessment
- `_analyze_rigor()`: GPT-powered rigor evaluation
- `_analyze_significance()`: GPT-powered significance analysis
- `_analyze_theory()`: GPT-powered theoretical contribution analysis
- `_analyze_practical()`: GPT-powered practical implications analysis

## Scoring System

### Weighted Scoring
The overall score is calculated using weighted averages:

```python
weights = {
    'novelty': 0.25,        # 25% - Novelty and originality
    'rigor': 0.20,          # 20% - Methodological rigor
    'significance': 0.20,   # 20% - Results significance
    'theory': 0.15,         # 15% - Theoretical contribution
    'practical': 0.10,      # 10% - Practical implications
    'impact': 0.05,         # 5% - Research impact
    'reproducibility': 0.05 # 5% - Reproducibility quality
}
```

### Score Interpretation
- **0.85-1.0**: Excellent scientific quality
- **0.70-0.84**: Good scientific quality
- **0.50-0.69**: Fair scientific quality
- **0.0-0.49**: Poor scientific quality

## Usage Examples

### Basic Usage
```python
from modules.check_quality import evaluate
from models.paper import Paper

# Load paper
paper = Paper(raw_text="Your paper text here...")

# Basic evaluation
result = evaluate(paper, use_gpt=False)

print(f"Score: {result['score']:.2f}")
print(f"Feedback: {result['feedback']}")
print(f"Breakdown: {result['score_breakdown']}")
```

### With GPT Enhancement
```python
# Enhanced evaluation with GPT
result = evaluate(paper, use_gpt=True)

# Check if GPT was used
if result.get('gpt_analysis', {}).get('used'):
    print("GPT analysis was performed")
    print(f"Cost: ${result['gpt_analysis']['cost_info']['cost_usd']:.4f}")
else:
    print("GPT analysis was skipped")
```

### Accessing Detailed Analysis
```python
result = evaluate(paper, use_gpt=False)

# Access detailed metrics
novelty_details = result['detailed_analysis']['novelty']
print(f"Novelty indicators found: {novelty_details['novelty_indicators_found']}")
print(f"Contribution statements: {novelty_details['contribution_statements']}")
print(f"Specific contributions: {novelty_details['specific_contributions']}")
```

## Output Structure

### Basic Result Structure
```python
{
    'pillar_name': 'Scientific Quality',
    'score': 0.85,  # Overall weighted score
    'feedback': 'Enhanced feedback with visual indicators...',
    'score_breakdown': {
        'novelty': 0.90,
        'rigor': 0.85,
        'significance': 0.80,
        'theory': 0.75,
        'practical': 0.85,
        'impact': 0.70,
        'reproducibility': 0.90
    },
    'detailed_analysis': {
        'novelty': {...},      # Detailed novelty metrics
        'rigor': {...},        # Detailed rigor metrics
        'significance': {...}, # Detailed significance metrics
        'theory': {...},       # Detailed theory metrics
        'practical': {...},    # Detailed practical metrics
        'impact': {...},       # Detailed impact metrics
        'reproducibility': {...} # Detailed reproducibility metrics
    },
    'gpt_analysis_data': {...}, # Data prepared for GPT analysis
    'gpt_analysis': {...}       # GPT analysis results (if used)
}
```

### GPT Analysis Structure (when used)
```python
{
    'gpt_analysis': {
        'used': True,
        'analysis': {
            'overall_score': 0.87,
            'sub_modules': {
                'novelty_assessment': {'score': 0.90, 'novel_contributions': [...]},
                'rigor_evaluation': {'score': 0.85, 'design_quality': '...'},
                'significance_analysis': {'score': 0.80, 'evidence_strength': '...'},
                'theoretical_contribution': {'score': 0.75, 'frameworks': [...]},
                'practical_impact': {'score': 0.85, 'applications': [...]}
            },
            'issues': [...],
            'suggestions': [...],
            'strengths': [...],
            'final_verdict': 'GOOD'
        },
        'cost_info': {
            'cost_usd': 0.0034,
            'input_tokens': 1250,
            'output_tokens': 450,
            'total_tokens': 1700
        },
        'model': 'gpt-4o-mini'
    }
}
```

## Configuration

### GPT Integration Setup
To enable GPT analysis, ensure you have:

1. **OpenAI API Key**: Set in environment variables
   ```bash
   export OPENAI_API_KEY="sk-your-key-here"
   ```

2. **Required Dependencies**:
   ```bash
   pip install openai python-dotenv
   ```

3. **Cost Considerations**:
   - Average cost per paper: $0.003-0.005
   - GPT analysis is only triggered when basic score < 0.7 or critical issues detected
   - Cost tracking and optimization built-in

### Environment Variables
```bash
# Required for GPT integration
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional: Custom configuration
QUALITY_ANALYSIS_THRESHOLD=0.7  # Threshold for GPT analysis
```

## Testing

The module includes comprehensive tests covering:

- Basic evaluation functionality
- All enhanced analysis functions
- GPT integration (when available)
- Edge cases and error handling
- Score consistency and validation

Run tests with:
```bash
python -m pytest tests/test_quality_module.py -v
```

## Performance Considerations

### Basic Analysis
- **Speed**: ~0.1-0.3 seconds per paper
- **Memory**: Minimal memory usage
- **Dependencies**: Only standard Python libraries

### GPT Analysis
- **Speed**: ~2-5 seconds per paper (including API calls)
- **Cost**: $0.003-0.005 per paper
- **Network**: Requires internet connection
- **Rate Limits**: Subject to OpenAI API rate limits

## Best Practices

### For Basic Analysis
1. Use `use_gpt=False` for fast, cost-free analysis
2. Review detailed analysis metrics for specific insights
3. Check score breakdown for targeted improvements

### For GPT Analysis
1. Use `use_gpt=True` for papers with scores < 0.7
2. Monitor costs using built-in cost tracking
3. Review GPT sub-module scores for detailed insights
4. Use GPT analysis sparingly for cost optimization

### For Integration
1. Always handle both basic and GPT results
2. Check for GPT availability before enabling
3. Implement proper error handling for API failures
4. Consider caching results for repeated analysis

## Troubleshooting

### Common Issues

1. **GPT Analysis Not Working**
   - Check OpenAI API key is set correctly
   - Verify internet connection
   - Check API rate limits and quotas

2. **Low Scores**
   - Review detailed analysis for specific issues
   - Check if paper has all required sections
   - Verify text parsing is working correctly

3. **Performance Issues**
   - Use basic analysis for large-scale processing
   - Implement caching for repeated analysis
   - Consider batch processing for GPT analysis

### Debug Mode
Enable debug logging to troubleshoot issues:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features
- Integration with citation databases for impact assessment
- Machine learning models for automated quality prediction
- Multi-language support for international papers
- Integration with journal-specific quality standards
- Advanced statistical analysis for result validation

### Contributing
To contribute to this module:
1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure backward compatibility
5. Consider performance implications

## Changelog

### Version 2.0 (Current)
- ✅ Enhanced evaluation metrics with 7 dimensions
- ✅ GPT-4o-mini integration for deep analysis
- ✅ Comprehensive score breakdown and detailed analytics
- ✅ Advanced pattern recognition and context awareness
- ✅ Cost-optimized GPT analysis with intelligent triggering
- ✅ Enhanced feedback with visual indicators
- ✅ Comprehensive test suite
- ✅ Detailed documentation

### Version 1.0 (Previous)
- Basic novelty, rigor, significance, theory, and practical evaluation
- Simple scoring system
- Basic feedback generation
