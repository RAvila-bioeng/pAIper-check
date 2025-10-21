"""
Linguistic Quality Control Module - Enhanced Version
Advanced NLP-based evaluation of academic paper linguistic quality.

This module provides comprehensive analysis of:
- Orthographic and grammatical accuracy
- Terminology consistency and domain-specific vocabulary
- Academic writing style and tone
- Multilingual support and language detection
- Advanced error detection and reporting

Architecture:
- LinguisticAnalyzer: Main orchestrator class
- OrthographicAnalyzer: Spelling and typography analysis
- GrammarAnalyzer: Grammar and syntax evaluation
- TerminologyAnalyzer: Domain-specific term consistency
- StyleAnalyzer: Academic tone and formality assessment
- Preprocessor: Text cleaning and normalization
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum

# Core dependencies (with graceful fallbacks)
try:
    import spacy
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Limited linguistic analysis will be performed.")

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    logging.warning("pyspellchecker not available. Basic spell checking will be used.")

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available. Language detection will be skipped.")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    logging.warning("textstat not available. Readability analysis will be limited.")

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Advanced similarity analysis will be limited.")

from models.score import PillarResult


class ErrorType(Enum):
    """Types of linguistic errors detected."""
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    PUNCTUATION = "punctuation"
    STYLE = "style"
    TERMINOLOGY = "terminology"
    CONSISTENCY = "consistency"
    FORMALITY = "formality"


@dataclass
class LinguisticError:
    """Represents a single linguistic error found in the text."""
    error_type: ErrorType
    text: str
    start_pos: int
    end_pos: int
    severity: float  # 0.0 (minor) to 1.0 (critical)
    suggested_correction: Optional[str] = None
    context: Optional[str] = None
    explanation: Optional[str] = None


@dataclass
class SectionAnalysis:
    """Analysis results for a single section of the paper."""
    section_name: str
    text: str
    errors: List[LinguisticError]
    readability_score: float
    formality_score: float
    terminology_consistency: float


class Preprocessor:
    """Text preprocessing and normalization utilities."""
    
    @staticmethod
    def clean_and_normalize(text: str) -> str:
        """Clean and normalize text for analysis."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common encoding issues
        text = text.replace('\u2019', "'")  # Smart apostrophe
        text = text.replace('\u2018', "'")  # Left single quote
        text = text.replace('\u201c', '"')  # Left double quote
        text = text.replace('\u201d', '"')  # Right double quote
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u2014', '--')  # Em dash
        
        return text.strip()
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text."""
        # Improved sentence splitting that handles academic text better
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect the primary language of the text."""
        if not LANGDETECT_AVAILABLE or not text.strip():
            return "unknown"
        
        try:
            # Use first 1000 characters for language detection to avoid errors with short text
            sample_text = text[:1000] if len(text) > 1000 else text
            if len(sample_text.strip()) < 10:
                return "unknown"
            return detect(sample_text)
        except LangDetectException:
            return "unknown"


class BaseAnalyzer(ABC):
    """Base class for all linguistic analyzers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[LinguisticError]:
        """Analyze text and return list of errors found."""
        pass


class OrthographicAnalyzer(BaseAnalyzer):
    """Analyzes spelling, typography, and basic orthographic issues."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.spell_checker = None
        self.common_academic_words = self._load_academic_vocabulary()
        
        if SPELLCHECKER_AVAILABLE:
            try:
                self.spell_checker = SpellChecker(language='en')
                # Add academic domain-specific words
                self.spell_checker.word_frequency.load_words(self.common_academic_words)
            except Exception as e:
                self.logger.warning(f"Could not initialize spell checker: {e}")
    
    def _load_academic_vocabulary(self) -> Set[str]:
        """Load common academic vocabulary that should not be flagged as errors."""
        academic_words = {
            'algorithm', 'analysis', 'application', 'approach', 'assessment', 'characteristics',
            'classification', 'comparison', 'computation', 'conclusion', 'configuration', 'correlation',
            'demonstration', 'distribution', 'evaluation', 'experiment', 'framework', 'implementation',
            'improvement', 'investigation', 'limitation', 'methodology', 'observation', 'optimization',
            'parameter', 'performance', 'probability', 'reliability', 'representation', 'validation',
            'verification', 'visualization', 'statistical', 'empirical', 'theoretical', 'experimental'
        }
        return academic_words
    
    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[LinguisticError]:
        """Analyze text for orthographic errors."""
        errors = []
        
        # Check for basic spelling errors if spell checker is available
        if self.spell_checker:
            errors.extend(self._check_spelling_errors(text))
        
        # Check for common typos and misspellings
        errors.extend(self._check_common_typos(text))
        
        # Check for capitalization issues
        errors.extend(self._check_capitalization_issues(text))
        
        return errors
    
    def _check_spelling_errors(self, text: str) -> List[LinguisticError]:
        """Check for spelling errors using spell checker in a more efficient way."""
        errors = []
        
        # Extract words, filter by length and known academic words
        words = re.findall(r'\b[A-Za-z]+\b', text)
        
        unique_words = {word.lower() for word in words if len(word) <= 50 and not (word.lower() in self.common_academic_words or (word[0].isupper() and len(word) > 1))}
        
        # Find all misspelled words in one go
        misspelled = self.spell_checker.unknown(unique_words)
        
        if not misspelled:
            return errors

        # Create a map of misspelled words to their suggestions
        correction_map = {word: self.spell_checker.correction(word) for word in misspelled}

        # Find all occurrences of misspelled words in the original text
        for word in misspelled:
            try:
                # Use regex to find all occurrences of the misspelled word as a whole word
                for match in re.finditer(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE):
                    original_word = match.group(0)
                    start_pos = match.start()
                    
                    error = LinguisticError(
                        error_type=ErrorType.SPELLING,
                        text=original_word,
                        start_pos=start_pos,
                        end_pos=start_pos + len(original_word),
                        severity=0.7,
                        suggested_correction=correction_map.get(word),
                        explanation=f"Possible spelling error: '{original_word}'"
                    )
                    errors.append(error)
            except re.error:
                # Ignore regex errors for complex/problematic word patterns
                continue
        
        return errors
    
    def _check_common_typos(self, text: str) -> List[LinguisticError]:
        """Check for common typos in academic writing."""
        errors = []
        
        common_typos = {
            r'\bteh\b': 'the',
            r'\badn\b': 'and',
            r'\bhte\b': 'the',
            r'\btaht\b': 'that',
            r'\bwih\b': 'with',
            r'\bthier\b': 'their',
            r'\brecieve\b': 'receive',
            r'\boccured\b': 'occurred',
            r'\bseperate\b': 'separate',
            r'\bdefinately\b': 'definitely',
            r'\bprinicple\b': 'principle',
            r'\benviroment\b': 'environment',
            r'\bexperiement\b': 'experiment',
            r'\bmethodolgy\b': 'methodology',
            r'\banaylsis\b': 'analysis'
        }
        
        for pattern, correction in common_typos.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                error = LinguisticError(
                    error_type=ErrorType.SPELLING,
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    severity=0.8,
                    suggested_correction=correction,
                    explanation=f"Common typo found: '{match.group()}'"
                )
                errors.append(error)
        
        return errors
    
    def _check_capitalization_issues(self, text: str) -> List[LinguisticError]:
        """Check for capitalization issues."""
        errors = []
        
        # Check for lowercase after periods (should be uppercase)
        for match in re.finditer(r'\.\s+[a-z]', text):
            error = LinguisticError(
                error_type=ErrorType.PUNCTUATION,
                text=match.group(),
                start_pos=match.start() + 2,
                end_pos=match.end(),
                severity=0.6,
                suggested_correction=match.group()[:-1] + match.group()[-1].upper(),
                explanation="Sentence should start with capital letter after period"
            )
            errors.append(error)
        
        return errors


class GrammarAnalyzer(BaseAnalyzer):
    """Analyzes grammar and syntax issues."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.nlp_model = None
        
        if SPACY_AVAILABLE:
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
                self.logger.debug("spaCy English model loaded successfully")
            except (OSError, IOError) as e:
                self.logger.warning(f"spaCy English model not found: {e}. Limited grammar analysis will be performed.")
            except Exception as e:
                self.logger.warning(f"Unexpected error loading spaCy model: {e}. Limited grammar analysis will be performed.")
    
    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[LinguisticError]:
        """Analyze text for grammar errors."""
        errors = []
        
        if self.nlp_model:
            errors.extend(self._analyze_with_spacy(text))
        else:
            errors.extend(self._analyze_with_regex(text))
        
        return errors
    
    def _analyze_with_spacy(self, text: str) -> List[LinguisticError]:
        """Use spaCy for advanced grammar analysis."""
        errors = []
        doc = self.nlp_model(text)
        
        # Check for sentence fragments
        for sent in doc.sents:
            if len([token for token in sent if not token.is_punct and not token.is_space]) < 3:
                error = LinguisticError(
                    error_type=ErrorType.GRAMMAR,
                    text=sent.text.strip(),
                    start_pos=sent.start_char,
                    end_pos=sent.end_char,
                    severity=0.5,
                    explanation="Potential sentence fragment"
                )
                errors.append(error)
        
        # Check for subject-verb agreement issues (basic)
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                if token.morph.get("Number") and token.head.morph.get("Number"):
                    subj_number = token.morph.get("Number")[0] if token.morph.get("Number") else None
                    verb_number = token.head.morph.get("Number")[0] if token.head.morph.get("Number") else None
                    if subj_number != verb_number:
                        error = LinguisticError(
                            error_type=ErrorType.GRAMMAR,
                            text=f"{token.text} {token.head.text}",
                            start_pos=token.idx,
                            end_pos=token.head.idx + len(token.head.text),
                            severity=0.8,
                            explanation="Potential subject-verb disagreement"
                        )
                        errors.append(error)
        
        return errors
    
    def _analyze_with_regex(self, text: str) -> List[LinguisticError]:
        """Fallback grammar analysis using regex patterns."""
        errors = []
        
        # Check for double spaces
        for match in re.finditer(r' {2,}', text):
            error = LinguisticError(
                error_type=ErrorType.PUNCTUATION,
                text=match.group(),
                start_pos=match.start(),
                end_pos=match.end(),
                severity=0.3,
                suggested_correction=" ",
                explanation="Excessive whitespace found"
            )
            errors.append(error)
        
        # Check for missing spaces after punctuation
        for match in re.finditer(r'[.!?][A-Za-z]', text):
            error = LinguisticError(
                error_type=ErrorType.PUNCTUATION,
                text=match.group(),
                start_pos=match.start() + 1,
                end_pos=match.end(),
                severity=0.6,
                explanation="Missing space after punctuation"
            )
            errors.append(error)
        
        return errors


class TerminologyAnalyzer(BaseAnalyzer):
    """Analyzes terminology consistency and domain-specific vocabulary."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.term_variations = self._load_term_variations()
        self.domain_terms = set()
    
    def _load_term_variations(self) -> Dict[str, Set[str]]:
        """Load common term variations that should be consistent."""
        return {
            'dataset': {'data set', 'data-set', 'datasets', 'data sets'},
            'machine learning': {'ml', 'machine-learning', 'ML'},
            'artificial intelligence': {'ai', 'A.I.', 'artificial-intelligence', 'AI'},
            'neural network': {'neural networks', 'neural-networks', 'NN', 'neural nets'},
            'deep learning': {'dl', 'deep-learning', 'DL'},
            'accuracy': {'accuracies', 'ACC', 'acc'},
            'precision': {'prec', 'PREC'},
            'recall': {'rec', 'REC'},
            'f-score': {'f1-score', 'f1_score', 'F1-score', 'F1_score'}
        }
    
    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[LinguisticError]:
        """Analyze text for terminology consistency issues."""
        errors = []
        
        # Extract and track term usage
        term_usage = {}
        
        for canonical_term, variations in self.term_variations.items():
            all_variants = variations | {canonical_term}
            
            for variant in all_variants:
                # Case-insensitive search
                pattern = r'\b' + re.escape(variant) + r'\b'
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                if matches:
                    if canonical_term not in term_usage:
                        term_usage[canonical_term] = []
                    term_usage[canonical_term].extend(matches)
        
        # Check for inconsistencies
        for canonical_term, matches in term_usage.items():
            if len(matches) > 1:
                # Check if multiple variants are used
                used_variants = set()
                for match in matches:
                    used_variants.add(match.group())
                
                if len(used_variants) > 1:
                    first_variant = used_variants.pop()
                    for variant in used_variants:
                        for match in matches:
                            if match.group() == variant:
                                error = LinguisticError(
                                    error_type=ErrorType.TERMINOLOGY,
                                    text=variant,
                                    start_pos=match.start(),
                                    end_pos=match.end(),
                                    severity=0.7,
                                    suggested_correction=first_variant,
                                    explanation=f"Inconsistent terminology. Consider using '{first_variant}' consistently."
                                )
                                errors.append(error)
        
        return errors


class StyleAnalyzer(BaseAnalyzer):
    """Analyzes academic writing style and formality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.informal_patterns = self._load_informal_patterns()
        self.academic_phrases = self._load_academic_phrases()
    
    def _load_informal_patterns(self) -> List[str]:
        """Load patterns of informal language to avoid."""
        return [
            r'\bawesome\b', r'\bcool\b', r'\bnice\b', r'\bgreat\b',
            r'\blots of\b', r'\btons of\b', r'\bpretty much\b',
            r'\bkind of\b', r'\bsort of\b', r'\bvery very\b',
            r'\breally really\b', r'\bso so\b', r'\bway too\b',
            r'\bcheck out\b', r'\blook at\b', r'\btake a look\b',
            r'\bget it\b', r'\bmake sense\b', r'\bgo ahead\b'
        ]
    
    def _load_academic_phrases(self) -> List[str]:
        """Load common academic phrases that indicate good style."""
        return [
            r'\bin order to\b', r'\bwith respect to\b', r'\bwith regard to\b',
            r'\bin terms of\b', r'\bin accordance with\b', r'\bsubsequent to\b',
            r'\bprior to\b', r'\bin conjunction with\b', r'\bby means of\b'
        ]
    
    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[LinguisticError]:
        """Analyze text for style and formality issues."""
        errors = []
        
        # Check for informal language
        errors.extend(self._check_informal_language(text))
        
        # Check for inappropriate contractions
        errors.extend(self._check_contractions(text))
        
        # Check for appropriate passive voice usage
        errors.extend(self._check_passive_voice(text))
        
        return errors
    
    def _check_informal_language(self, text: str) -> List[LinguisticError]:
        """Check for informal language patterns."""
        errors = []
        
        for pattern in self.informal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                error = LinguisticError(
                    error_type=ErrorType.STYLE,
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    severity=0.6,
                    explanation="Informal language detected in academic context"
                )
                errors.append(error)
        
        return errors
    
    def _check_contractions(self, text: str) -> List[LinguisticError]:
        """Check for inappropriate contractions in academic writing."""
        errors = []
        
        contractions = [
            r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b", r"\bwon't\b",
            r"\bcan't\b", r"\bcouldn't\b", r"\bshouldn't\b", r"\bwouldn't\b",
            r"\bit's\b", r"\bthat's\b", r"\bthere's\b", r"\bhere's\b"
        ]
        
        for pattern in contractions:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                error = LinguisticError(
                    error_type=ErrorType.STYLE,
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    severity=0.5,
                    explanation="Contractions are generally avoided in formal academic writing"
                )
                errors.append(error)
        
        return errors
    
    def _check_passive_voice(self, text: str) -> List[LinguisticError]:
        """Check for appropriate use of passive voice in academic writing."""
        # This is more nuanced - passive voice is often appropriate in academic writing
        # We'll check for excessive use rather than flagging all instances
        
        passive_patterns = [
            r'\bis\s+\w+ed\b', r'\bwas\s+\w+ed\b', r'\bwere\s+\w+ed\b',
            r'\bhas\s+been\s+\w+ed\b', r'\bhave\s+been\s+\w+ed\b'
        ]
        
        passive_count = sum(len(re.findall(pattern, text)) for pattern in passive_patterns)
        total_sentences = len(re.split(r'[.!?]+', text))
        
        # Flag if more than 70% of sentences use passive voice (might be excessive)
        if total_sentences > 0 and passive_count / total_sentences > 0.7:
            return [LinguisticError(
                error_type=ErrorType.STYLE,
                text="[Multiple instances]",
                start_pos=0,
                end_pos=len(text),
                severity=0.3,
                explanation="Excessive use of passive voice. Consider using active voice where appropriate."
            )]
        
        return []


class ReadabilityAnalyzer:
    """Analyzes text readability and complexity."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_readability(self, text: str) -> Dict[str, float]:
        """Analyze readability metrics."""
        if not TEXTSTAT_AVAILABLE or not text:
            return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}
        
        try:
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "gunning_fog": textstat.gunning_fog(text),
                "automated_readability_index": textstat.automated_readability_index(text)
            }
        except Exception as e:
            self.logger.warning(f"Error calculating readability metrics: {e}")
            return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}


class LinguisticAnalyzer:
    """Main orchestrator for linguistic quality analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize analyzers
        self.orthographic_analyzer = OrthographicAnalyzer(config)
        self.grammar_analyzer = GrammarAnalyzer(config)
        self.terminology_analyzer = TerminologyAnalyzer(config)
        self.style_analyzer = StyleAnalyzer(config)
        self.readability_analyzer = ReadabilityAnalyzer()
        
        # Initialize preprocessor
        self.preprocessor = Preprocessor()
    
    def analyze(self, paper) -> Dict[str, Any]:
        """
        Perform comprehensive linguistic analysis of the paper.
        
        Args:
            paper: Paper object containing the text to analyze
            
        Returns:
            Dict containing analysis results
        """
        if not paper or not paper.full_text:
            return self._create_empty_result()
        
        # Preprocess text
        text = self.preprocessor.clean_and_normalize(paper.full_text)
        
        # Detect language
        detected_language = self.preprocessor.detect_language(text)
        
        # Analyze different aspects
        all_errors = []
        
        # Orthographic analysis
        orthographic_errors = self.orthographic_analyzer.analyze(text)
        all_errors.extend(orthographic_errors)
        
        # Grammar analysis
        grammar_errors = self.grammar_analyzer.analyze(text)
        all_errors.extend(grammar_errors)
        
        # Terminology analysis
        terminology_errors = self.terminology_analyzer.analyze(text, {"paper": paper})
        all_errors.extend(terminology_errors)
        
        # Style analysis
        style_errors = self.style_analyzer.analyze(text)
        all_errors.extend(style_errors)
        
        # Calculate scores
        overall_score = self._calculate_overall_score(all_errors, text)
        
        # Analyze readability
        readability_metrics = self.readability_analyzer.analyze_readability(text)
        
        # Analyze by sections if available
        section_analyses = []
        if hasattr(paper, 'sections') and paper.sections:
            for section in paper.sections:
                if section.content:
                    section_text = self.preprocessor.clean_and_normalize(section.content)
                    section_errors = self._analyze_section(section_text)
                    section_analysis = SectionAnalysis(
                        section_name=section.title,
                        text=section_text,
                        errors=section_errors,
                        readability_score=readability_metrics.get("flesch_reading_ease", 0.0),
                        formality_score=self._calculate_formality_score(section_text),
                        terminology_consistency=self._calculate_terminology_consistency(section_text)
                    )
                    section_analyses.append(section_analysis)
        
        # Generate detailed feedback
        feedback = self._generate_comprehensive_feedback(all_errors, readability_metrics, detected_language)
        
        return {
            "pillar_name": "Linguistic Quality",
            "score": overall_score,
            "feedback": feedback,
            "detailed_analysis": {
                "total_errors": len(all_errors),
                "error_types": self._categorize_errors(all_errors),
                "readability_metrics": readability_metrics,
                "detected_language": detected_language,
                "section_analyses": [
                    {
                        "section_name": sa.section_name,
                        "error_count": len(sa.errors),
                        "readability_score": sa.readability_score,
                        "formality_score": sa.formality_score,
                        "terminology_consistency": sa.terminology_consistency
                    } for sa in section_analyses
                ],
                "specific_errors": [
                    {
                        "type": error.error_type.value,
                        "text": error.text,
                        "severity": error.severity,
                        "suggestion": error.suggested_correction,
                        "explanation": error.explanation
                    } for error in all_errors[:20]  # Limit to top 20 errors
                ]
            }
        }
    
    def _analyze_section(self, section_text: str) -> List[LinguisticError]:
        """Analyze a single section of the paper."""
        errors = []
        errors.extend(self.orthographic_analyzer.analyze(section_text))
        errors.extend(self.grammar_analyzer.analyze(section_text))
        errors.extend(self.style_analyzer.analyze(section_text))
        return errors
    
    def _calculate_overall_score(self, errors: List[LinguisticError], text: str) -> float:
        """Calculate overall linguistic quality score."""
        if not text:
            return 0.0
        
        # Base score starts at 1.0
        score = 1.0
        
        # Deduct points based on error severity and frequency
        total_error_impact = sum(error.severity for error in errors)
        text_length = len(text.split())  # Word count
        
        # Normalize error impact by text length
        if text_length > 0:
            normalized_impact = total_error_impact / (text_length / 100)  # Errors per 100 words
            score -= min(0.8, normalized_impact * 0.1)  # Cap maximum deduction at 0.8
        
        # Bonus for good readability (academic papers should be moderately complex)
        if TEXTSTAT_AVAILABLE:
            try:
                reading_ease = textstat.flesch_reading_ease(text)
                if 20 <= reading_ease <= 60:  # Good range for academic papers
                    score = min(1.0, score + 0.1)
            except Exception:
                pass
        
        return max(0.0, score)
    
    def _calculate_formality_score(self, text: str) -> float:
        """Calculate formality score based on style indicators."""
        if not text:
            return 0.0
        
        formal_indicators = len(re.findall(r'\bin order to\b|\bwith respect to\b|\baccordingly\b', text, re.IGNORECASE))
        informal_indicators = len(re.findall(r'\bawesome\b|\bcool\b|\bnice\b', text, re.IGNORECASE))
        contractions = len(re.findall(r"\b\w+'t\b|\b\w+'s\b", text))
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        formality_score = (formal_indicators - informal_indicators - contractions * 0.5) / (word_count / 100)
        return max(0.0, min(1.0, (formality_score + 2) / 4))  # Normalize to 0-1 range
    
    def _calculate_terminology_consistency(self, text: str) -> float:
        """Calculate terminology consistency score."""
        if not text:
            return 1.0
        
        # This would be enhanced with the terminology analyzer results
        # For now, return a basic score based on term variation patterns
        inconsistencies = self.terminology_analyzer.analyze(text)
        error_count = len(inconsistencies)
        word_count = len(text.split())
        
        if word_count == 0:
            return 1.0
        
        consistency_score = max(0.0, 1.0 - (error_count / (word_count / 200)))
        return consistency_score
    
    def _categorize_errors(self, errors: List[LinguisticError]) -> Dict[str, int]:
        """Categorize errors by type."""
        categories = {}
        for error in errors:
            error_type = error.error_type.value
            categories[error_type] = categories.get(error_type, 0) + 1
        return categories
    
    def _generate_comprehensive_feedback(self, errors: List[LinguisticError], 
                                       readability_metrics: Dict[str, float],
                                       detected_language: str) -> str:
        """Generate comprehensive feedback based on analysis."""
        feedback_parts = []
        
        if not errors:
            feedback_parts.append("Excellent linguistic quality with no significant errors detected.")
        else:
            error_categories = self._categorize_errors(errors)
            
            if ErrorType.SPELLING.value in error_categories:
                count = error_categories[ErrorType.SPELLING.value]
                feedback_parts.append(f"Found {count} spelling error{'s' if count > 1 else ''}. Consider thorough proofreading.")
            
            if ErrorType.GRAMMAR.value in error_categories:
                count = error_categories[ErrorType.GRAMMAR.value]
                feedback_parts.append(f"Detected {count} grammar issue{'s' if count > 1 else ''}. Review sentence structure and syntax.")
            
            if ErrorType.TERMINOLOGY.value in error_categories:
                count = error_categories[ErrorType.TERMINOLOGY.value]
                feedback_parts.append(f"Found {count} terminology inconsistency{'ies' if count > 1 else ''}. Ensure consistent use of technical terms.")
            
            if ErrorType.STYLE.value in error_categories:
                count = error_categories[ErrorType.STYLE.value]
                feedback_parts.append(f"Identified {count} style issue{'s' if count > 1 else ''}. Maintain formal academic tone throughout.")
        
        # Add readability feedback
        if readability_metrics:
            reading_ease = readability_metrics.get("flesch_reading_ease", 0)
            if reading_ease > 70:
                feedback_parts.append("Text may be too simple for academic audience. Consider more sophisticated vocabulary.")
            elif reading_ease < 20:
                feedback_parts.append("Text may be overly complex. Consider simplifying some sentences for better readability.")
        
        # Add language detection feedback
        if detected_language not in ["en", "unknown"]:
            feedback_parts.append(f"Primary language detected as {detected_language}. Ensure English formatting and style guidelines are followed.")
        
        return " ".join(feedback_parts) if feedback_parts else "Good linguistic quality overall."
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result for papers with no text."""
        return {
            "pillar_name": "Linguistic Quality",
            "score": 0.0,
            "feedback": "No text available for linguistic analysis.",
            "detailed_analysis": {
                "total_errors": 0,
                "error_types": {},
                "readability_metrics": {},
                "detected_language": "unknown"
            }
        }


# Main evaluation function (maintains compatibility with existing system)
def evaluate(paper) -> dict:
    """
    Evaluate linguistic quality of the paper.
    
    Args:
        paper: Paper object with text content
        
    Returns:
        dict: Score and feedback for linguistic quality
        
    This function maintains compatibility with the existing evaluation system
    while providing enhanced linguistic analysis capabilities.
    """
    analyzer = LinguisticAnalyzer()
    result = analyzer.analyze(paper)
    
    # Convert to PillarResult format for compatibility
    return PillarResult(
        pillar_name=result["pillar_name"],
        score=result["score"],
        feedback=result["feedback"],
        gpt_analysis_data=result.get("detailed_analysis")
    ).__dict__