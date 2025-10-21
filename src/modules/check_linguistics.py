"""
Linguistic Quality Control Module - Optimized for English Academic Papers
Advanced NLP-based evaluation focusing on language-level aspects of academic research papers.

This module provides comprehensive analysis of:
- Orthographic and grammatical accuracy
- Academic writing style and tone consistency
- Lexical precision and redundancy detection
- General linguistic quality index for scientific writing

Key Features:
- Optimized for English academic/scientific papers
- No multilingual processing (English-only focus)
- Scientific complexity-aware (no penalties for complex scientific language)
- Comprehensive error detection with specific academic context
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any, Counter
from enum import Enum
from collections import defaultdict

# Core dependencies (with graceful fallbacks)
try:
    import spacy
    from spacy.tokens import Doc, Span, Token
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
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    logging.warning("textstat not available. Readability analysis will be limited.")

try:
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("numpy not available. Advanced analysis will be limited.")

from models.score import PillarResult


class ErrorType(Enum):
    """Types of linguistic errors detected in academic papers."""
    ORTHOGRAPHIC = "orthographic"  # Spelling and typography
    GRAMMATICAL = "grammatical"    # Grammar and syntax
    PUNCTUATION = "punctuation"    # Punctuation and formatting
    STYLE = "style"               # Academic style and formality
    LEXICAL = "lexical"           # Word choice and precision
    REDUNDANCY = "redundancy"     # Unnecessary repetition
    TONE = "tone"                 # Academic tone consistency


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
    confidence: float = 1.0  # Confidence in the error detection


@dataclass
class LinguisticMetrics:
    """Comprehensive linguistic quality metrics."""
    orthographic_score: float
    grammatical_score: float
    style_score: float
    lexical_precision_score: float
    redundancy_score: float
    tone_consistency_score: float
    overall_quality_index: float
    
    # Detailed statistics
    total_errors: int
    error_distribution: Dict[str, int]
    word_count: int
    sentence_count: int
    academic_vocabulary_ratio: float


class AcademicVocabulary:
    """Manages academic and scientific vocabulary for context-aware analysis."""
    
    def __init__(self):
        self.scientific_terms = self._load_scientific_vocabulary()
        self.academic_phrases = self._load_academic_phrases()
        self.complex_acceptable = self._load_complex_acceptable_terms()
    
    def _load_scientific_vocabulary(self) -> Set[str]:
        """Load comprehensive scientific vocabulary."""
        return {
            # General academic terms
            'analyze', 'analysis', 'analyzed', 'analing', 'approach', 'assessment', 'characteristics',
            'classification', 'comprehensive', 'conclusion', 'configuration', 'demonstration',
            'determination', 'development', 'evaluation', 'examination', 'experimental', 'framework',
            'hypothesis', 'implementation', 'investigation', 'measurement', 'methodology', 'observation',
            'optimization', 'parameter', 'performance', 'probability', 'procedure', 'protocol',
            'quantification', 'relationship', 'reliability', 'reproducibility', 'resolution',
            'significance', 'simulation', 'specification', 'statistical', 'substantiation',
            'synthesis', 'validation', 'variability', 'verification', 'visualization',
            
            # Scientific domains
            'algorithm', 'biosensor', 'biomarker', 'chromatography', 'electrochemical',
            'fluorescence', 'immunoassay', 'kinetics', 'microarray', 'nanotechnology',
            'photoluminescence', 'polymerase', 'spectroscopy', 'thermodynamics',
            
            # Mathematical and statistical
            'coefficient', 'correlation', 'distribution', 'regression', 'variance',
            'standard deviation', 'confidence interval', 'p-value', 'significance level',
            
            # Research methodology
            'concurrent', 'longitudinal', 'prospective', 'retrospective', 'randomized',
            'controlled', 'double-blind', 'placebo-controlled', 'cross-sectional'
        }
    
    def _load_academic_phrases(self) -> List[str]:
        """Load common academic phrases that indicate good academic style."""
        return [
            r'\bin order to\b', r'\bwith respect to\b', r'\bwith regard to\b',
            r'\bin terms of\b', r'\bin accordance with\b', r'\bsubsequent to\b',
            r'\bprior to\b', r'\bin conjunction with\b', r'\bby means of\b',
            r'\bit is worth noting\b', r'\bit should be noted\b', r'\bit is important to\b',
            r'\bthe results suggest\b', r'\bthe findings indicate\b', r'\bthe data demonstrate\b'
        ]
    
    def _load_complex_acceptable_terms(self) -> Set[str]:
        """Load terms that are complex but acceptable in scientific writing."""
        return {
            'methodology', 'biocompatibility', 'electrochemically', 'spectrophotometrically',
            'immunohistochemically', 'pharmacokinetically', 'thermodynamically',
            'electrophysiologically', 'immunofluorescently', 'morphometrically'
        }
    
    def is_academic_term(self, word: str) -> bool:
        """Check if a word is a recognized academic/scientific term."""
        return word.lower() in self.scientific_terms
    
    def is_complex_acceptable(self, word: str) -> bool:
        """Check if a complex word is acceptable in scientific context."""
        return word.lower() in self.complex_acceptable
    
    def count_academic_phrases(self, text: str) -> int:
        """Count occurrences of academic phrases in text."""
        count = 0
        for phrase_pattern in self.academic_phrases:
            count += len(re.findall(phrase_pattern, text, re.IGNORECASE))
        return count


class TextPreprocessor:
    """Advanced text preprocessing for academic papers."""
    
    @staticmethod
    def clean_academic_text(text: str) -> str:
        """Clean and normalize academic text while preserving scientific formatting."""
        if not text:
            return ""
        
        # Remove excessive whitespace while preserving paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Fix common encoding issues in academic papers
        replacements = {
            '\u2019': "'",  # Smart apostrophe
            '\u2018': "'",  # Left single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...', # Ellipsis
        }
        
        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)
        
        return text.strip()
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences with improved handling for academic writing."""
        # Split on periods, exclamation marks, and question marks
        # but be careful with abbreviations and decimal numbers
        sentences = re.split(r'(?<!\b[a-zA-Z])(?<!\d)\.(?!\d)(?![a-zA-Z])|!|\?', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    @staticmethod
    def extract_words(text: str) -> List[str]:
        """Extract words preserving hyphenated scientific terms."""
        # This regex preserves hyphenated terms common in scientific writing
        words = re.findall(r'\b[A-Za-z][A-Za-z-]*[A-Za-z]\b|\b[A-Za-z]\b', text)
        return [word for word in words if len(word) > 1]


class BaseAnalyzer(ABC):
    """Base class for all linguistic analyzers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.academic_vocab = AcademicVocabulary()
    
    @abstractmethod
    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[LinguisticError]:
        """Analyze text and return list of errors found."""
        pass


class OrthographicAnalyzer(BaseAnalyzer):
    """Advanced orthographic analysis for academic papers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.spell_checker = None
        self.academic_vocab = AcademicVocabulary()
        
        if SPELLCHECKER_AVAILABLE:
            try:
                self.spell_checker = SpellChecker(language='en')
                # Add scientific terms to the dictionary
                self.spell_checker.word_frequency.load_words(self.academic_vocab.scientific_terms)
            except Exception as e:
                self.logger.warning(f"Could not initialize spell checker: {e}")
    
    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[LinguisticError]:
        """Analyze text for orthographic errors."""
        errors = []
        
        # Check spelling errors with improved efficiency
        if self.spell_checker:
            errors.extend(self._check_spelling_errors(text))
        
        # Check for common academic typos
        errors.extend(self._check_academic_typos(text))
        
        # Check capitalization in academic context
        errors.extend(self._check_academic_capitalization(text))
        
        return errors
    
    def _check_spelling_errors(self, text: str) -> List[LinguisticError]:
        """Check for spelling errors using spell checker in a more efficient way."""
        errors = []
        
        # Extract words, filter by length and known academic words
        words = re.findall(r'\b[A-Za-z]+\b', text)
        
        unique_words = {word.lower() for word in words if len(word) <= 50 and not (
            word.lower() in self.academic_vocab.scientific_terms or 
            (word[0].isupper() and len(word) > 1) or
            self.academic_vocab.is_complex_acceptable(word.lower())
        )}
        
        # Find all misspelled words in one go
        if not unique_words or not self.spell_checker:
            return errors
            
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
                        error_type=ErrorType.ORTHOGRAPHIC,
                        text=original_word,
                        start_pos=start_pos,
                        end_pos=start_pos + len(original_word),
                        severity=0.8,
                        suggested_correction=correction_map.get(word),
                        explanation=f"Spelling error detected: '{original_word}'"
                    )
                    errors.append(error)
            except re.error:
                # Ignore regex errors for complex/problematic word patterns
                continue
        
        return errors
    
    def _check_academic_typos(self, text: str) -> List[LinguisticError]:
        """Check for common typos specific to academic writing."""
        errors = []
        
        academic_typos = {
            r'\bprinicple\b': 'principle',
            r'\benviroment\b': 'environment',
            r'\bexperiement\b': 'experiment',
            r'\bmethodolgy\b': 'methodology',
            r'\banaylsis\b': 'analysis',
            r'\bhypothsis\b': 'hypothesis',
            r'\bmethdology\b': 'methodology',
            r'\bperfomance\b': 'performance',
            r'\bparamter\b': 'parameter',
            r'\bstatitical\b': 'statistical',
            r'\bcorreltion\b': 'correlation',
            r'\bdistrubution\b': 'distribution'
        }
        
        for pattern, correction in academic_typos.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                error = LinguisticError(
                    error_type=ErrorType.ORTHOGRAPHIC,
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    severity=0.9,
                    suggested_correction=correction,
                    explanation=f"Common academic typo: '{match.group()}'"
                )
                errors.append(error)
        
        return errors
    
    def _check_academic_capitalization(self, text: str) -> List[LinguisticError]:
        """Check for capitalization issues in academic context."""
        errors = []
        
        # Check for lowercase after periods (should be uppercase in new sentences)
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
    """Advanced grammar analysis using spaCy for academic writing."""
    
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
        
        if self.nlp_model and len(text.strip()) > 10:
            errors.extend(self._analyze_with_spacy(text))
    else:
            errors.extend(self._analyze_with_heuristics(text))
        
        return errors
    
    def _analyze_with_spacy(self, text: str) -> List[LinguisticError]:
        """Use spaCy for advanced grammar analysis."""
        errors = []
        
        try:
            doc = self.nlp_model(text)
            
            # Check for subject-verb agreement
            errors.extend(self._check_subject_verb_agreement(doc))
            
            # Check for sentence fragments (but be lenient for scientific writing)
            errors.extend(self._check_sentence_fragments(doc))
            
            # Check for common grammatical issues in academic writing
            errors.extend(self._check_academic_grammar_issues(doc))
            
        except Exception as e:
            self.logger.warning(f"Error in spaCy analysis: {e}")
            
        return errors
    
    def _check_subject_verb_agreement(self, doc: Doc) -> List[LinguisticError]:
        """Check for subject-verb agreement issues."""
        errors = []
        
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                # Check number agreement
                if hasattr(token, 'morph') and hasattr(token.head, 'morph'):
                    subj_number = token.morph.get("Number")
                    verb_number = token.head.morph.get("Number")
                    
                    if subj_number and verb_number and subj_number != verb_number:
                        error = LinguisticError(
                            error_type=ErrorType.GRAMMATICAL,
                            text=f"{token.text} {token.head.text}",
                            start_pos=token.idx,
                            end_pos=token.head.idx + len(token.head.text),
                            severity=0.8,
                            explanation="Potential subject-verb disagreement"
                        )
                        errors.append(error)
        
        return errors
    
    def _check_sentence_fragments(self, doc: Doc) -> List[LinguisticError]:
        """Check for sentence fragments, being lenient with scientific writing."""
        errors = []
        
        for sent in doc.sents:
            # Count meaningful tokens (excluding punctuation and stop words)
            meaningful_tokens = [token for token in sent 
                              if not token.is_punct and not token.is_space 
                              and not token.is_stop and token.pos_ != "PUNCT"]
            
            # Be more lenient with scientific writing - some technical phrases might be acceptable
            if len(meaningful_tokens) < 3 and not any(self.academic_vocab.is_academic_term(token.text.lower()) 
                                                     for token in meaningful_tokens):
                error = LinguisticError(
                    error_type=ErrorType.GRAMMATICAL,
                    text=sent.text.strip(),
                    start_pos=sent.start_char,
                    end_pos=sent.end_char,
                    severity=0.5,
                    explanation="Potential sentence fragment - may be acceptable in technical context"
                )
                errors.append(error)
        
        return errors
    
    def _check_academic_grammar_issues(self, doc: Doc) -> List[LinguisticError]:
        """Check for specific grammar issues common in academic writing."""
        errors = []
        
        # Check for missing articles before singular nouns in academic context
        for token in doc:
            if (token.pos_ == "NOUN" and 
                token.morph.get("Number") == ["Sing"] and
                not any(ancestor.dep_ == "det" for ancestor in token.ancestors) and
                not token.text.lower() in ["data", "research", "analysis"]):  # Some nouns don't need articles
                
                # Only flag if it's not a compound noun or scientific term
                if not (token.head.pos_ == "NOUN" or 
                       self.academic_vocab.is_academic_term(token.text.lower())):
                    continue  # Skip this case
                
                error = LinguisticError(
                    error_type=ErrorType.GRAMMATICAL,
                    text=token.text,
                    start_pos=token.idx,
                    end_pos=token.idx + len(token.text),
                    severity=0.4,
                    explanation="Consider adding article before singular noun"
                )
                errors.append(error)
        
        return errors
    
    def _analyze_with_heuristics(self, text: str) -> List[LinguisticError]:
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


class AcademicStyleAnalyzer(BaseAnalyzer):
    """Analyzes academic writing style and tone consistency."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.informal_patterns = self._load_informal_patterns()
        
    def _load_informal_patterns(self) -> List[str]:
        """Load patterns of informal language inappropriate for academic writing."""
        return [
            r'\bawesome\b', r'\bcool\b', r'\bnice work\b', r'\bgreat job\b',
            r'\blots of\b', r'\btons of\b', r'\bpretty much\b',
            r'\bkind of\b', r'\bsort of\b', r'\bvery very\b',
            r'\breally really\b', r'\bso so\b', r'\bway too\b',
            r'\bcheck out\b', r'\btake a look\b', r'\bget it\b',
            r'\bmake sense\b', r'\bgo ahead\b', r'\byeah\b', r'\byup\b'
        ]
    
    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[LinguisticError]:
        """Analyze text for academic style and tone issues."""
        errors = []
        
        errors.extend(self._check_informal_language(text))
        errors.extend(self._check_contractions(text))
        errors.extend(self._check_academic_tone_consistency(text))
        errors.extend(self._check_voice_consistency(text))
        
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
                    severity=0.7,
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
            r"\bit's\b", r"\bthat's\b", r"\bthere's\b", r"\bhere's\b",
            r"\bwe're\b", r"\bthey're\b", r"\byou're\b"
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
    
    def _check_academic_tone_consistency(self, text: str) -> List[LinguisticError]:
        """Check for consistent academic tone throughout."""
        errors = []
        
        # Count academic phrases indicating good style
        academic_phrase_count = self.academic_vocab.count_academic_phrases(text)
        
        # Count sentences
        sentences = TextPreprocessor.extract_sentences(text)
        sentence_count = len(sentences)
        
        # If there are many sentences but few academic phrases, flag potential issues
        if sentence_count > 10 and academic_phrase_count < sentence_count * 0.1:
            error = LinguisticError(
                error_type=ErrorType.TONE,
                text="[Multiple instances]",
                start_pos=0,
                end_pos=len(text),
                severity=0.3,
                explanation="Consider using more formal academic phrasing throughout the text"
            )
            errors.append(error)
        
        return errors
    
    def _check_voice_consistency(self, text: str) -> List[LinguisticError]:
        """Check for appropriate use of active/passive voice in academic writing."""
        # This is more nuanced - academic writing often uses passive voice appropriately
        # We'll check for extreme overuse rather than flagging all instances
        
    passive_patterns = [
        r'\bis\s+\w+ed\b', r'\bwas\s+\w+ed\b', r'\bwere\s+\w+ed\b',
        r'\bhas\s+been\s+\w+ed\b', r'\bhave\s+been\s+\w+ed\b'
    ]
    
    passive_count = sum(len(re.findall(pattern, text)) for pattern in passive_patterns)
        total_sentences = len(TextPreprocessor.extract_sentences(text))
        
        # Flag if more than 80% of sentences use passive voice (might be excessive)
        if total_sentences > 0 and passive_count / total_sentences > 0.8:
            return [LinguisticError(
                error_type=ErrorType.STYLE,
                text="[Multiple instances]",
                start_pos=0,
                end_pos=len(text),
                severity=0.4,
                explanation="Excessive use of passive voice. Consider using active voice where appropriate for clarity."
            )]
        
        return []


class LexicalPrecisionAnalyzer(BaseAnalyzer):
    """Analyzes lexical precision and word choice in academic writing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vague_terms = self._load_vague_terms()
        self.precision_issues = self._load_precision_issues()
    
    def _load_vague_terms(self) -> Set[str]:
        """Load terms that may be too vague for precise academic writing."""
        return {
            'thing', 'stuff', 'things', 'lot', 'lots', 'many', 'much', 'some',
            'very', 'really', 'quite', 'rather', 'somewhat', 'kind of', 'sort of'
        }
    
    def _load_precision_issues(self) -> Dict[str, str]:
        """Load common precision issues and better alternatives."""
        return {
            'data': 'dataset',  # Context-dependent
            'result': 'finding',  # More specific for research
            'good': 'effective',  # More precise
            'bad': 'ineffective',  # More precise
            'big': 'large/significant',  # More academic
            'small': 'minor/minimal',  # More precise
        }
    
    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[LinguisticError]:
        """Analyze text for lexical precision issues."""
        errors = []
        
        errors.extend(self._check_vague_terminology(text))
        errors.extend(self._check_precision_issues(text))
        errors.extend(self._check_redundancy(text))
        
        return errors
    
    def _check_vague_terminology(self, text: str) -> List[LinguisticError]:
        """Check for overly vague terminology."""
        errors = []
        
        words = TextPreprocessor.extract_words(text.lower())
        
        for word in words:
            if word in self.vague_terms:
                # Find the original case occurrence
                pattern = r'\b' + re.escape(word) + r'\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    error = LinguisticError(
                        error_type=ErrorType.LEXICAL,
                        text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        severity=0.4,
                        explanation=f"Consider using more precise terminology instead of '{match.group()}'"
                    )
                    errors.append(error)
        
        return errors
    
    def _check_precision_issues(self, text: str) -> List[LinguisticError]:
        """Check for precision issues and suggest improvements."""
        errors = []
        
        for vague_term, suggestion in self.precision_issues.items():
            pattern = r'\b' + re.escape(vague_term) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                error = LinguisticError(
                    error_type=ErrorType.LEXICAL,
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    severity=0.3,
                    suggested_correction=suggestion,
                    explanation=f"Consider using more precise term: '{suggestion}' instead of '{match.group()}'"
                )
                errors.append(error)
        
        return errors
    
    def _check_redundancy(self, text: str) -> List[LinguisticError]:
        """Check for redundant expressions and unnecessary repetition."""
        errors = []
        
        redundant_patterns = {
            r'\babsolutely\s+essential\b': 'essential',
            r'\bcompletely\s+full\b': 'full',
            r'\bcompletely\s+empty\b': 'empty',
            r'\btotally\s+unique\b': 'unique',
            r'\bvery\s+unique\b': 'unique',
            r'\badvance\s+planning\b': 'planning',
            r'\bboth\s+of\s+the\b': 'both',
            r'\bend\s+result\b': 'result',
            r'\bfirst\s+introduction\b': 'introduction'
        }
        
        for pattern, replacement in redundant_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                error = LinguisticError(
                    error_type=ErrorType.REDUNDANCY,
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    severity=0.5,
                    suggested_correction=replacement,
                    explanation=f"Redundant expression: '{match.group()}' can be simplified"
                )
                errors.append(error)
        
        return errors


class LinguisticAnalyzer:
    """Main orchestrator for comprehensive linguistic quality analysis of academic papers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize specialized analyzers
        self.orthographic_analyzer = OrthographicAnalyzer(config)
        self.grammar_analyzer = GrammarAnalyzer(config)
        self.style_analyzer = AcademicStyleAnalyzer(config)
        self.lexical_analyzer = LexicalPrecisionAnalyzer(config)
        
        # Initialize utilities
        self.preprocessor = TextPreprocessor()
        self.academic_vocab = AcademicVocabulary()
    
    def analyze(self, paper) -> Dict[str, Any]:
        """
        Perform comprehensive linguistic analysis of the academic paper.
        
        Args:
            paper: Paper object containing the text to analyze
            
        Returns:
            Dict containing comprehensive analysis results
        """
        if not paper or not paper.full_text:
            return self._create_empty_result()
        
        # Preprocess text
        text = self.preprocessor.clean_academic_text(paper.full_text)
        words = TextPreprocessor.extract_words(text)
        sentences = TextPreprocessor.extract_sentences(text)
        
        if not text or len(words) < 10:
            return self._create_empty_result()
        
        # Perform all analyses
        all_errors = []
        all_errors.extend(self.orthographic_analyzer.analyze(text))
        all_errors.extend(self.grammar_analyzer.analyze(text))
        all_errors.extend(self.style_analyzer.analyze(text))
        all_errors.extend(self.lexical_analyzer.analyze(text))
        
        # Calculate comprehensive metrics
        metrics = self._calculate_linguistic_metrics(all_errors, text, words, sentences)
        
        # Generate detailed feedback
        feedback = self._generate_comprehensive_feedback(all_errors, metrics)
        
        return {
            "pillar_name": "Linguistic Quality",
            "score": metrics.overall_quality_index,
            "feedback": feedback,
            "detailed_analysis": {
                "metrics": {
                    "orthographic_score": metrics.orthographic_score,
                    "grammatical_score": metrics.grammatical_score,
                    "style_score": metrics.style_score,
                    "lexical_precision_score": metrics.lexical_precision_score,
                    "redundancy_score": metrics.redundancy_score,
                    "tone_consistency_score": metrics.tone_consistency_score,
                    "overall_quality_index": metrics.overall_quality_index
                },
                "statistics": {
                    "total_errors": metrics.total_errors,
                    "error_distribution": metrics.error_distribution,
                    "word_count": metrics.word_count,
                    "sentence_count": metrics.sentence_count,
                    "academic_vocabulary_ratio": metrics.academic_vocabulary_ratio
                },
                "top_errors": [
                    {
                        "type": error.error_type.value,
                        "text": error.text,
                        "severity": error.severity,
                        "suggestion": error.suggested_correction,
                        "explanation": error.explanation,
                        "confidence": error.confidence
                    } for error in sorted(all_errors, key=lambda x: x.severity, reverse=True)[:10]
                ]
            }
        }
    
    def _calculate_linguistic_metrics(self, errors: List[LinguisticError], 
                                    text: str, words: List[str], sentences: List[str]) -> LinguisticMetrics:
        """Calculate comprehensive linguistic quality metrics."""
        
        # Categorize errors by type
        error_by_type = defaultdict(list)
        for error in errors:
            error_by_type[error.error_type].append(error)
        
        # Calculate scores for each aspect (higher is better)
        orthographic_score = max(0.0, 1.0 - (len(error_by_type[ErrorType.ORTHOGRAPHIC]) * 0.15))
        grammatical_score = max(0.0, 1.0 - (len(error_by_type[ErrorType.GRAMMATICAL]) * 0.12))
        style_score = max(0.0, 1.0 - (len(error_by_type[ErrorType.STYLE]) * 0.08 + len(error_by_type[ErrorType.TONE]) * 0.1))
        lexical_score = max(0.0, 1.0 - (len(error_by_type[ErrorType.LEXICAL]) * 0.06))
        redundancy_score = max(0.0, 1.0 - (len(error_by_type[ErrorType.REDUNDANCY]) * 0.05))
        
        # Normalize by text length to be fair to longer papers
        word_count = len(words)
        if word_count > 0:
            error_density = len(errors) / (word_count / 100)  # Errors per 100 words
        
        # Calculate academic vocabulary ratio
        academic_words = sum(1 for word in words if self.academic_vocab.is_academic_term(word.lower()))
        academic_vocab_ratio = academic_words / word_count if word_count > 0 else 0.0
        
        # Calculate overall quality index with scientific complexity bonus
        overall_score = (
            orthographic_score * 0.25 +
            grammatical_score * 0.25 +
            style_score * 0.20 +
            lexical_score * 0.15 +
            redundancy_score * 0.15
        )
        
        # Bonus for appropriate academic vocabulary usage (no penalty for complex scientific language)
        if academic_vocab_ratio > 0.15:  # Good academic vocabulary usage
            overall_score = min(1.0, overall_score + 0.1)
        
        # Create error distribution
        error_distribution = {error_type.value: len(errors_list) 
                            for error_type, errors_list in error_by_type.items()}
        
        return LinguisticMetrics(
            orthographic_score=orthographic_score,
            grammatical_score=grammatical_score,
            style_score=style_score,
            lexical_precision_score=lexical_score,
            redundancy_score=redundancy_score,
            tone_consistency_score=style_score,  # Using style score as proxy
            overall_quality_index=max(0.0, overall_score),
            total_errors=len(errors),
            error_distribution=error_distribution,
            word_count=word_count,
            sentence_count=len(sentences),
            academic_vocabulary_ratio=academic_vocab_ratio
        )
    
    def _generate_comprehensive_feedback(self, errors: List[LinguisticError], 
                                       metrics: LinguisticMetrics) -> str:
        """Generate comprehensive and actionable feedback."""
    feedback_parts = []
    
        if metrics.total_errors == 0:
            feedback_parts.append("Excellent linguistic quality with no significant errors detected.")
        else:
            # Provide specific feedback based on error types and severity
            if metrics.orthographic_score < 0.8:
                ortho_count = metrics.error_distribution.get('orthographic', 0)
                feedback_parts.append(f"Found {ortho_count} orthographic error{'s' if ortho_count != 1 else ''}. Consider thorough proofreading and spell-checking.")
            
            if metrics.grammatical_score < 0.8:
                grammar_count = metrics.error_distribution.get('grammatical', 0)
                feedback_parts.append(f"Detected {grammar_count} grammatical issue{'s' if grammar_count != 1 else ''}. Review sentence structure and syntax.")
            
            if metrics.style_score < 0.8:
                style_count = metrics.error_distribution.get('style', 0) + metrics.error_distribution.get('tone', 0)
                feedback_parts.append(f"Identified {style_count} style issue{'s' if style_count != 1 else ''}. Maintain consistent academic tone and avoid informal language.")
            
            if metrics.lexical_precision_score < 0.8:
                lexical_count = metrics.error_distribution.get('lexical', 0)
                feedback_parts.append(f"Found {lexical_count} lexical precision issue{'s' if lexical_count != 1 else ''}. Consider more precise terminology.")
            
            if metrics.redundancy_score < 0.8:
                redundancy_count = metrics.error_distribution.get('redundancy', 0)
                feedback_parts.append(f"Detected {redundancy_count} redundant expression{'s' if redundancy_count != 1 else ''}. Consider removing unnecessary repetition.")
        
        # Add positive feedback for good aspects
        if metrics.academic_vocabulary_ratio > 0.15:
            feedback_parts.append("Good use of appropriate academic vocabulary throughout the text.")
        
        # Overall assessment
        if metrics.overall_quality_index >= 0.8:
            feedback_parts.append("Overall linguistic quality is strong.")
        elif metrics.overall_quality_index >= 0.6:
            feedback_parts.append("Linguistic quality is acceptable but could be improved.")
        else:
            feedback_parts.append("Significant linguistic issues require attention before publication.")
        
        return " ".join(feedback_parts) if feedback_parts else "Linguistic analysis completed."
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result for papers with no text."""
        return {
            "pillar_name": "Linguistic Quality",
            "score": 0.0,
            "feedback": "No text available for linguistic analysis.",
            "detailed_analysis": {
                "metrics": {},
                "statistics": {"total_errors": 0, "word_count": 0, "sentence_count": 0},
                "top_errors": []
            }
        }


# Main evaluation function (maintains compatibility with existing system)
def evaluate(paper) -> dict:
    """
    Evaluate linguistic quality of the academic paper.
    
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