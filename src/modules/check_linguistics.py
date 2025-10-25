"""
Linguistic Quality Control Module - Enhanced Version
NLP-based evaluation with detailed error location reporting.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from collections import defaultdict

# Core dependencies with graceful fallbacks
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Using basic analysis.")

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    logging.warning("pyspellchecker not available.")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    logging.warning("textstat not available.")


class ErrorType(Enum):
    """Types of linguistic errors detected."""
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    PUNCTUATION = "punctuation"
    STYLE = "style"
    TERMINOLOGY = "terminology"


@dataclass
class LinguisticError:
    """Represents a single linguistic error with location."""
    error_type: ErrorType
    text: str
    severity: float  # 0.0 to 1.0
    position: int  # Character position in text
    line_number: int  # Line number (approximate)
    context: str  # Surrounding text for context
    suggestion: Optional[str] = None
    explanation: Optional[str] = None
<<<<<<< HEAD
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
=======
>>>>>>> origin/monica


class LinguisticAnalyzer:
    """Main orchestrator for linguistic quality analysis."""
    
    # Academic vocabulary to whitelist
    ACADEMIC_WORDS = {
        'algorithm', 'analysis', 'application', 'approach', 'dataset',
        'evaluation', 'experiment', 'framework', 'implementation', 'methodology',
        'optimization', 'parameter', 'performance', 'validation', 'visualization',
        'neural', 'computational', 'statistical', 'empirical', 'theoretical',
        'hypothesis', 'correlation', 'regression', 'significant', 'variance',
        'coefficient', 'matrix', 'vector', 'tensor', 'gradient', 'convergence'
    }
    
    # Common typos mapping
    COMMON_TYPOS = {
        r'\bteh\b': 'the',
        r'\badn\b': 'and',
        r'\brecieve\b': 'receive',
        r'\boccured\b': 'occurred',
        r'\bseperate\b': 'separate',
        r'\bdefinately\b': 'definitely',
        r'\benviroment\b': 'environment',
        r'\bexperiement\b': 'experiment',
        r'\bperformence\b': 'performance',
        r'\baccuracy\b': 'accuracy'
    }
    
    # Informal patterns to avoid
    INFORMAL_PATTERNS = [
        r'\bawesome\b', r'\bcool\b', r'\bnice\b', r'\blots of\b',
        r'\bpretty much\b', r'\bkind of\b', r'\bsort of\b',
        r'\bstuff\b', r'\bthing\b', r'\bget\b'
    ]
    
    # Contractions to avoid in formal writing
    CONTRACTIONS = [
        r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b", r"\bwon't\b",
        r"\bcan't\b", r"\bcouldn't\b", r"\bit's\b", r"\bthat's\b",
        r"\bwouldn't\b", r"\bshouldn't\b", r"\bisn't\b", r"\baren't\b"
    ]
    
    # Terminology variations to check for consistency
    TERM_VARIATIONS = {
        'dataset': {'data set', 'data-set'},
        'machine learning': {'ml', 'machine-learning'},
        'neural network': {'neural networks', 'neural-network'},
        'deep learning': {'dl', 'deep-learning'}
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_spell_checker()
        self._init_nlp()
        
    def _init_spell_checker(self):
        """Initialize spell checker if available."""
        self.spell_checker = None
        if SPELLCHECKER_AVAILABLE:
            try:
                self.spell_checker = SpellChecker(language='en')
                self.spell_checker.word_frequency.load_words(self.ACADEMIC_WORDS)
            except Exception as e:
                self.logger.warning(f"Spell checker init failed: {e}")
    
    def _init_nlp(self):
        """Initialize spaCy if available."""
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'textcat'])
                self.logger.info("spaCy loaded successfully")
            except Exception as e:
                self.logger.warning(f"spaCy load failed: {e}")
    
    def _get_context(self, text: str, position: int, context_chars: int = 60) -> str:
        """Extract context around an error position."""
        start = max(0, position - context_chars)
        end = min(len(text), position + context_chars)
        context = text[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context.strip()
    
    def _get_line_number(self, text: str, position: int) -> int:
        """Calculate line number for a given character position."""
        return text[:position].count('\n') + 1
    
    def analyze(self, paper) -> Dict[str, Any]:
        """
        Perform comprehensive linguistic analysis.
        
        Args:
            paper: Paper object containing text
            
        Returns:
            Dict with score, feedback, and detailed error locations
        """
        if not paper or not paper.full_text:
            return self._create_empty_result()
        
        text = self._clean_text(paper.full_text)
        
        # Limit analysis to reasonable length to avoid performance issues
        max_length = 50000  # ~50k characters
        truncated = False
        if len(text) > max_length:
            self.logger.info(f"Text truncated from {len(text)} to {max_length} chars")
            text = text[:max_length]
            truncated = True
        
        # Collect all errors
        errors = []
        
        try:
            # Spelling analysis
            errors.extend(self._check_spelling(text))
            
            # Grammar analysis
            errors.extend(self._check_grammar(text))
            
            # Style analysis
            errors.extend(self._check_style(text))
            
            # Terminology consistency
            errors.extend(self._check_terminology(text))
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            return self._create_error_result(str(e))
        
        # Sort errors by position for better readability
        errors.sort(key=lambda e: e.position)
        
        # Calculate scores
        score = self._calculate_score(errors, text)
        feedback = self._generate_feedback(errors, text, truncated)
        error_details = self._format_error_details(errors)
        
        return {
            "pillar_name": "Linguistic Quality",
            "score": score,
            "feedback": feedback,
            "total_errors": len(errors),
            "error_breakdown": self._categorize_errors(errors),
            "error_details": error_details,
            "readability": self._analyze_readability(text),
            "text_truncated": truncated
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Fix common encoding issues
        replacements = {
            '\u2019': "'", '\u2018': "'",
            '\u201c': '"', '\u201d': '"',
            '\u2013': '-', '\u2014': '--'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Normalize whitespace but keep newlines for line numbering
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def _check_spelling(self, text: str) -> List[LinguisticError]:
        """Check for spelling errors with location."""
        errors = []
        
        # First check common typos (fast)
        for pattern, correction in self.COMMON_TYPOS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                errors.append(LinguisticError(
                    error_type=ErrorType.SPELLING,
                    text=match.group(),
                    severity=0.8,
                    position=match.start(),
                    line_number=self._get_line_number(text, match.start()),
                    context=self._get_context(text, match.start()),
                    suggestion=correction,
                    explanation=f"Common typo: '{match.group()}' should be '{correction}'"
                ))
        
        # Then use spell checker if available
        if self.spell_checker and len(text) < 30000:
            try:
                # Extract words with their positions
                word_pattern = re.compile(r'\b[a-z]{3,15}\b')
                word_matches = list(word_pattern.finditer(text.lower()))
                
                # Get unique words to check
                unique_words = {match.group() for match in word_matches 
                               if match.group() not in self.ACADEMIC_WORDS}
                
                # Limit to avoid long processing
                unique_words = list(unique_words)[:100]
                
                # Check spelling in batch
                misspelled = self.spell_checker.unknown(unique_words)
                
                # Add errors with positions (limit to first 20)
                for word in list(misspelled)[:20]:
                    # Find first occurrence of this misspelled word
                    for match in word_matches:
                        if match.group() == word:
                            correction = self.spell_checker.correction(word)
                            errors.append(LinguisticError(
                                error_type=ErrorType.SPELLING,
                                text=word,
                                severity=0.6,
                                position=match.start(),
                                line_number=self._get_line_number(text, match.start()),
                                context=self._get_context(text, match.start()),
                                suggestion=correction,
                                explanation=f"Possible spelling error"
                            ))
                            break  # Only report first occurrence
            except Exception as e:
                self.logger.debug(f"Spell check error: {e}")
        
        return errors
    
    def _check_grammar(self, text: str) -> List[LinguisticError]:
        """Check for grammar issues with location."""
        errors = []
        
        # Double spaces
        for match in re.finditer(r'  +', text):
            errors.append(LinguisticError(
                error_type=ErrorType.PUNCTUATION,
                text="[double space]",
                severity=0.3,
                position=match.start(),
                line_number=self._get_line_number(text, match.start()),
                context=self._get_context(text, match.start()),
                suggestion=" ",
                explanation="Extra whitespace"
            ))
        
        # Missing space after punctuation
        for match in re.finditer(r'[.!?,][A-Za-z]', text):
            errors.append(LinguisticError(
                error_type=ErrorType.PUNCTUATION,
                text=match.group(),
                severity=0.5,
                position=match.start(),
                line_number=self._get_line_number(text, match.start()),
                context=self._get_context(text, match.start()),
                explanation="Missing space after punctuation"
            ))
        
        # Lowercase after period (excluding abbreviations)
        for match in re.finditer(r'\.\s+[a-z]', text):
            # Skip common abbreviations
            before = text[max(0, match.start()-3):match.start()]
            if not re.search(r'\b(e\.g|i\.e|etc|Dr|Mr|Mrs|vs)$', before):
                errors.append(LinguisticError(
                    error_type=ErrorType.GRAMMAR,
                    text=match.group(),
                    severity=0.6,
                    position=match.start(),
                    line_number=self._get_line_number(text, match.start()),
                    context=self._get_context(text, match.start()),
                    explanation="Sentence should start with capital letter"
                ))
        
        # Advanced grammar with spaCy (if available)
        if self.nlp and len(text) < 20000:
            try:
                doc = self.nlp(text[:20000])
                
                # Check for very short sentences (fragments)
                for sent in doc.sents:
                    words = [t for t in sent if not t.is_punct and not t.is_space]
                    if len(words) < 3 and len(sent.text) > 10:
                        errors.append(LinguisticError(
                            error_type=ErrorType.GRAMMAR,
                            text=sent.text[:50],
                            severity=0.4,
                            position=sent.start_char,
                            line_number=self._get_line_number(text, sent.start_char),
                            context=self._get_context(text, sent.start_char),
                            explanation="Possible sentence fragment"
                        ))
            except Exception as e:
                self.logger.debug(f"spaCy grammar check error: {e}")
        
        return errors
    
    def _check_style(self, text: str) -> List[LinguisticError]:
        """Check for style issues with location."""
        errors = []
        
        # Informal language
        for pattern in self.INFORMAL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                errors.append(LinguisticError(
                    error_type=ErrorType.STYLE,
                    text=match.group(),
                    severity=0.5,
                    position=match.start(),
                    line_number=self._get_line_number(text, match.start()),
                    context=self._get_context(text, match.start()),
                    explanation="Informal language in academic context"
                ))
        
        # Contractions
        for pattern in self.CONTRACTIONS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                expanded = self._expand_contraction(match.group())
                errors.append(LinguisticError(
                    error_type=ErrorType.STYLE,
                    text=match.group(),
                    severity=0.4,
                    position=match.start(),
                    line_number=self._get_line_number(text, match.start()),
                    context=self._get_context(text, match.start()),
                    suggestion=expanded,
                    explanation="Avoid contractions in formal writing"
                ))
        
        return errors
    
    def _expand_contraction(self, contraction: str) -> str:
        """Expand common contractions."""
        expansions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "can't": "cannot", "couldn't": "could not",
            "it's": "it is", "that's": "that is", "wouldn't": "would not",
            "shouldn't": "should not", "isn't": "is not", "aren't": "are not"
        }
        return expansions.get(contraction.lower(), contraction)
    
    def _check_terminology(self, text: str) -> List[LinguisticError]:
        """Check terminology consistency with location."""
        errors = []
        
        for canonical, variations in self.TERM_VARIATIONS.items():
            all_forms = variations | {canonical}
            found_forms = {}
            
            for form in all_forms:
                pattern = r'\b' + re.escape(form) + r'\b'
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    found_forms[form] = matches[0]  # Store first occurrence
            
            # If multiple forms used, flag inconsistency
            if len(found_forms) > 1:
                first_match = list(found_forms.values())[0]
                errors.append(LinguisticError(
                    error_type=ErrorType.TERMINOLOGY,
                    text=f"{canonical} variations",
                    severity=0.6,
                    position=first_match.start(),
                    line_number=self._get_line_number(text, first_match.start()),
                    context=f"Multiple forms found: {', '.join(found_forms.keys())}",
                    suggestion=f"Use '{canonical}' consistently",
                    explanation=f"Inconsistent terminology: {', '.join(found_forms.keys())}"
                ))
        
        return errors
    
    def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics - ADJUSTED FOR ACADEMIC CONTEXT."""
        if not TEXTSTAT_AVAILABLE or not text:
            return {}
        
        try:
            flesch = textstat.flesch_reading_ease(text)
            grade = textstat.flesch_kincaid_grade(text)
            
            # Add academic context interpretation
            interpretation = ""
            if 0 <= flesch <= 30:
                interpretation = "Very difficult (appropriate for scientific papers)"
            elif 30 < flesch <= 50:
                interpretation = "Difficult (typical for academic research)"
            elif 50 < flesch <= 60:
                interpretation = "Fairly difficult (acceptable for research)"
            else:
                interpretation = "May be too simple for academic context"
            
            return {
                "flesch_reading_ease": flesch,
                "flesch_kincaid_grade": grade,
                "interpretation": interpretation
            }
        except Exception as e:
            self.logger.debug(f"Readability calculation error: {e}")
            return {}
    
    def _calculate_score(self, errors: List[LinguisticError], text: str) -> float:
        """Calculate overall linguistic quality score."""
        if not text:
            return 0.0
        
        # Start with perfect score
        score = 1.0
        
        # Calculate error impact
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        # Weight errors by severity
        total_severity = sum(e.severity for e in errors)
        
        # Normalize by text length (per 100 words)
        normalized_impact = (total_severity / (word_count / 100)) * 0.1
        
        # Deduct from score
        score -= min(0.85, normalized_impact)
        
        return max(0.0, score)
    
    def _categorize_errors(self, errors: List[LinguisticError]) -> Dict[str, int]:
        """Count errors by type."""
        categories = defaultdict(int)
        for error in errors:
            categories[error.error_type.value] += 1
        return dict(categories)
    
<<<<<<< HEAD
    def _generate_comprehensive_feedback(self, errors: List[LinguisticError], 
                                       metrics: LinguisticMetrics) -> str:
        """Generate comprehensive and actionable feedback."""
        feedback_parts = []
=======
    def _format_error_details(self, errors: List[LinguisticError]) -> List[Dict[str, Any]]:
        """Format errors for detailed output."""
        return [
            {
                "type": error.error_type.value,
                "text": error.text,
                "line": error.line_number,
                "position": error.position,
                "severity": error.severity,
                "context": error.context,
                "suggestion": error.suggestion,
                "explanation": error.explanation
            }
            for error in errors
        ]
>>>>>>> origin/monica
    
    def _generate_feedback(self, errors: List[LinguisticError], text: str, truncated: bool) -> str:
        """Generate human-readable feedback - ADJUSTED FOR ACADEMIC CONTEXT."""
        if not errors:
            return "Excellent linguistic quality with no significant issues detected."
        
        parts = []
        error_cats = self._categorize_errors(errors)
        
        # Spelling feedback
        if 'spelling' in error_cats:
            count = error_cats['spelling']
            parts.append(f"Found {count} potential spelling issue{'s' if count > 1 else ''}. "
                        "Review the error details section for specific locations.")
        
        # Grammar feedback
        if 'grammar' in error_cats or 'punctuation' in error_cats:
            grammar_count = error_cats.get('grammar', 0)
            punct_count = error_cats.get('punctuation', 0)
            total = grammar_count + punct_count
            parts.append(f"Detected {total} grammar/punctuation issue{'s' if total > 1 else ''}. "
                        "Check the error details for line numbers and suggestions.")
        
        # Style feedback
        if 'style' in error_cats:
            count = error_cats['style']
            parts.append(f"Found {count} style issue{'s' if count > 1 else ''}. "
                        "Consider maintaining formal academic tone throughout.")
        
        # Terminology feedback
        if 'terminology' in error_cats:
            count = error_cats['terminology']
            parts.append(f"Detected {count} terminology inconsistency{'ies' if count > 1 else ''}. "
                        "Use technical terms consistently.")
        
        # Readability feedback - ADJUSTED FOR ACADEMIC CONTEXT
        if TEXTSTAT_AVAILABLE and text:
            try:
                ease = textstat.flesch_reading_ease(text)
                # Only warn if text is TOO simple for academic work
                if ease > 60:
                    parts.append("Text readability may be too simple for a scientific paper. "
                               "Consider using more technical language where appropriate.")
                # Do NOT criticize complexity - it's expected in academic papers
            except Exception:
                pass
        
        if truncated:
            parts.append("Note: Analysis was performed on the first 50,000 characters of the document.")
        
        feedback = " ".join(parts) if parts else "Minor issues detected. Overall quality is good."
        feedback += "\n\nSee 'error_details' for specific locations and suggestions."
        
        return feedback
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create result for empty papers."""
        return {
            "pillar_name": "Linguistic Quality",
            "score": 0.0,
            "feedback": "No text available for analysis.",
            "total_errors": 0,
            "error_breakdown": {},
            "error_details": [],
            "readability": {},
            "text_truncated": False
        }
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create result when analysis fails."""
        return {
            "pillar_name": "Linguistic Quality",
            "score": 0.5,
            "feedback": f"Analysis encountered errors: {error_msg}. Partial results only.",
            "total_errors": 0,
            "error_breakdown": {},
            "error_details": [],
            "readability": {},
            "text_truncated": False
        }


def evaluate(paper) -> dict:
    """
    Main evaluation function for compatibility with existing system.
    
    Args:
        paper: Paper object with text content
        
    Returns:
        dict: Score and detailed feedback for linguistic quality
    """
    analyzer = LinguisticAnalyzer()
    return analyzer.analyze(paper)