# Linguistic Quality Control Module - Enhanced Documentation

## Overview

The enhanced Linguistic Quality Control module provides comprehensive analysis of academic papers' linguistic quality using advanced NLP techniques. Unlike the previous basic implementation that only used regex patterns, this version offers production-ready analysis with sophisticated error detection and reporting.

## Key Features

### Core Capabilities
- **Advanced Spelling & Orthography Analysis**: Uses spaCy and spell-checking libraries for comprehensive error detection
- **Grammar & Syntax Analysis**: Deep analysis of sentence structure, subject-verb agreement, and syntax issues
- **Terminology Consistency**: Ensures consistent use of domain-specific terms throughout the paper
- **Academic Style Assessment**: Evaluates formality, tone, and adherence to academic writing standards
- **Multilingual Support**: Language detection and appropriate analysis based on detected language
- **Readability Analysis**: Comprehensive readability metrics using established academic standards

### Technical Improvements
- **Modular Architecture**: Clean separation of concerns with specialized analyzer classes
- **Error Classification**: Detailed error categorization with severity levels and specific suggestions
- **Graceful Fallbacks**: System continues to work even if some NLP libraries are unavailable
- **Performance Optimized**: Efficient text processing with minimal memory overhead
- **Extensible Design**: Easy to add new analysis capabilities or customize existing ones

## Architecture

### Core Components

#### 1. LinguisticAnalyzer (Main Orchestrator)
- Coordinates all analysis components
- Calculates overall quality scores
- Generates comprehensive feedback

#### 2. Specialized Analyzers

**OrthographicAnalyzer**
- Detects spelling errors using advanced spell-checking
- Identifies common typos in academic writing
- Checks capitalization and typography issues

**GrammarAnalyzer**
- Uses spaCy for advanced grammar analysis
- Detects sentence fragments and syntax issues
- Identifies subject-verb agreement problems

**TerminologyAnalyzer**
- Ensures consistent use of technical terms
- Detects variations in domain-specific vocabulary
- Provides suggestions for standardization

**StyleAnalyzer**
- Evaluates academic writing style and formality
- Detects inappropriate contractions and informal language
- Analyzes passive voice usage patterns

#### 3. Support Classes

**Preprocessor**
- Text cleaning and normalization
- Sentence extraction and parsing
- Language detection capabilities

**ReadabilityAnalyzer**
- Calculates readability metrics (Flesch-Kincaid, Gunning Fog, etc.)
- Provides academic writing complexity assessment

**Error Classification**
- `LinguisticError` dataclass for detailed error representation
- `ErrorType` enum for consistent error categorization
- Severity scoring and suggestion system

## Usage

### Basic Usage
```python
from modules.check_linguistics import evaluate
from models.paper import Paper

# Evaluate a paper (maintains compatibility with existing system)
result = evaluate(paper)
print(f"Score: {result['score']}")
print(f"Feedback: {result['feedback']}")
```

### Advanced Usage
```python
from modules.check_linguistics import LinguisticAnalyzer

# Create analyzer with custom configuration
config = {
    'spelling_strictness': 0.8,
    'grammar_check_level': 'advanced'
}
analyzer = LinguisticAnalyzer(config)

# Perform comprehensive analysis
result = analyzer.analyze(paper)

# Access detailed error information
for error in result['detailed_analysis']['specific_errors']:
    print(f"Error: {error['text']} - {error['explanation']}")
```

## Dependencies

### Required Libraries
- `spacy>=3.7.0`: Advanced NLP processing
- `langdetect>=1.0.9`: Language detection
- `textstat>=0.7.3`: Readability analysis
- `pyspellchecker>=0.6.2`: Spelling error detection
- `scikit-learn>=1.3.0`: Advanced text similarity analysis
- `numpy>=1.24.0`: Numerical computations

### Optional Libraries
- `language-tool-python>=2.7.1`: Additional grammar checking (if available)

### Installation Requirements
After installing the new requirements, you'll need to download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

## Error Types and Scoring

### Error Classification
1. **SPELLING**: Orthographic and typography errors
2. **GRAMMAR**: Syntax and grammatical structure issues
3. **PUNCTUATION**: Punctuation and formatting problems
4. **STYLE**: Academic writing style and formality issues
5. **TERMINOLOGY**: Inconsistent use of domain-specific terms
6. **CONSISTENCY**: General consistency issues
7. **FORMALITY**: Level of formality appropriate for academic writing

### Scoring Algorithm
- Base score starts at 1.0
- Deductions based on error severity and frequency
- Normalization by text length (errors per 100 words)
- Bonuses for appropriate academic complexity
- Final score range: 0.0 to 1.0

### Severity Levels
- **0.0-0.3**: Minor issues (e.g., formatting)
- **0.4-0.6**: Moderate issues (e.g., style, informal language)
- **0.7-0.9**: Major issues (e.g., grammar, spelling errors)
- **1.0**: Critical issues (e.g., severe grammatical errors)

## Configuration Options

### Available Configuration Parameters
```python
config = {
    'spelling_strictness': 0.8,      # How strict spelling checking should be
    'grammar_check_level': 'advanced', # Level of grammar analysis
    'style_formality_threshold': 0.7,  # Threshold for academic formality
    'terminology_consistency_weight': 0.3, # Weight for terminology consistency
    'readability_target_range': [20, 60]  # Target reading ease score
}
```

## Integration Notes

### Backward Compatibility
The module maintains full backward compatibility with the existing system. The main `evaluate()` function continues to work as before, returning results in the expected `PillarResult` format.

### Performance Considerations
- Text preprocessing is optimized for academic papers
- Analyzers use lazy loading to minimize startup time
- Results are cached when possible to improve performance
- Graceful degradation when advanced libraries are unavailable

### Error Handling
- Comprehensive error handling ensures system stability
- Fallback mechanisms for missing dependencies
- Detailed logging for debugging and monitoring

## Future Enhancements

### Planned Improvements
1. **Domain-Specific Analysis**: Custom analyzers for different academic fields
2. **Advanced Grammar Checking**: Integration with more sophisticated grammar tools
3. **Citation Style Analysis**: Verification of citation formatting consistency
4. **Cross-Reference Validation**: Checking for consistent technical term usage across sections
5. **Plagiarism Detection**: Integration with plagiarism detection algorithms

### Extensibility Points
- Easy addition of new analyzer classes
- Configurable error detection thresholds
- Customizable feedback generation
- Plugin architecture for specialized analysis tools

This enhanced module represents a significant improvement over the basic regex-based approach, providing production-ready linguistic analysis capabilities that align with the project's goal of ethical and scientific evaluation of academic papers.
