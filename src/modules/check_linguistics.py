"""
Linguistic Quality Control Module - Optimized Version
Efficient NLP-based evaluation of academic paper linguistic quality.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any
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
    """Represents a single linguistic error."""
    error_type: ErrorType
    text: str
    severity: float  # 0.0 to 1.0
    suggestion: Optional[str] = None
    explanation: Optional[str] = None


class LinguisticAnalyzer:
    """Main orchestrator for linguistic quality analysis."""
    
    # Academic vocabulary to whitelist
    ACADEMIC_WORDS = {
        'algorithm', 'analysis', 'application', 'approach', 'dataset',
        'evaluation', 'experiment', 'framework', 'implementation', 'methodology',
        'optimization', 'parameter', 'performance', 'validation', 'visualization',
        'neural', 'computational', 'statistical', 'empirical', 'theoretical'
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
        r'\bexperiement\b': 'experiment'
    }
    
    # Informal patterns to avoid
    INFORMAL_PATTERNS = [
        r'\bawesome\b', r'\bcool\b', r'\bnice\b', r'\blots of\b',
        r'\bpretty much\b', r'\bkind of\b', r'\bsort of\b'
    ]
    
    # Contractions to avoid in formal writing
    CONTRACTIONS = [
        r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b", r"\bwon't\b",
        r"\bcan't\b", r"\bcouldn't\b", r"\bit's\b", r"\bthat's\b"
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
    
    def analyze(self, paper) -> Dict[str, Any]:
        """
        Perform comprehensive linguistic analysis.
        
        Args:
            paper: Paper object containing text
            
        Returns:
            Dict with score, feedback, and analysis details
        """
        if not paper or not paper.full_text:
            return self._create_empty_result()
        
        text = self._clean_text(paper.full_text)
        
        # Limit analysis to reasonable length to avoid performance issues
        max_length = 50000  # ~50k characters
        if len(text) > max_length:
            self.logger.info(f"Text truncated from {len(text)} to {max_length} chars")
            text = text[:max_length]
        
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
        
        # Calculate scores
        score = self._calculate_score(errors, text)
        feedback = self._generate_feedback(errors, text)
        
        return {
            "pillar_name": "Linguistic Quality",
            "score": score,
            "feedback": feedback,
            "total_errors": len(errors),
            "error_breakdown": self._categorize_errors(errors),
            "readability": self._analyze_readability(text)
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
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _check_spelling(self, text: str) -> List[LinguisticError]:
        """Check for spelling errors - OPTIMIZED."""
        errors = []
        
        # First check common typos (fast)
        for pattern, correction in self.COMMON_TYPOS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                errors.append(LinguisticError(
                    error_type=ErrorType.SPELLING,
                    text=match.group(),
                    severity=0.8,
                    suggestion=correction,
                    explanation=f"Common typo: '{match.group()}'"
                ))
        
        # Then use spell checker if available
        if self.spell_checker and len(text) < 30000:  # Only for shorter texts
            try:
                # Extract unique words efficiently
                words = set(re.findall(r'\b[a-z]{3,15}\b', text.lower()))
                
                # Filter out academic words and proper nouns
                words_to_check = {w for w in words 
                                 if w not in self.ACADEMIC_WORDS 
                                 and not w[0].isupper()}
                
                # Limit to 100 most suspicious words to avoid long processing
                words_to_check = list(words_to_check)[:100]
                
                # Check spelling in batch
                misspelled = self.spell_checker.unknown(words_to_check)
                
                # Add errors (limit to first 20 occurrences)
                for word in list(misspelled)[:20]:
                    correction = self.spell_checker.correction(word)
                    errors.append(LinguisticError(
                        error_type=ErrorType.SPELLING,
                        text=word,
                        severity=0.6,
                        suggestion=correction,
                        explanation=f"Possible spelling error"
                    ))
            except Exception as e:
                self.logger.debug(f"Spell check error: {e}")
        
        return errors
    
    def _check_grammar(self, text: str) -> List[LinguisticError]:
        """Check for grammar issues."""
        errors = []
        
        # Basic punctuation checks
        # Double spaces
        for match in re.finditer(r'  +', text):
            errors.append(LinguisticError(
                error_type=ErrorType.PUNCTUATION,
                text="[double space]",
                severity=0.3,
                suggestion=" ",
                explanation="Extra whitespace"
            ))
        
        # Missing space after punctuation
        for match in re.finditer(r'[.!?,][A-Za-z]', text):
            errors.append(LinguisticError(
                error_type=ErrorType.PUNCTUATION,
                text=match.group(),
                severity=0.5,
                explanation="Missing space after punctuation"
            ))
        
        # Lowercase after period
        for match in re.finditer(r'\.\s+[a-z]', text):
            errors.append(LinguisticError(
                error_type=ErrorType.GRAMMAR,
                text=match.group(),
                severity=0.6,
                explanation="Sentence should start with capital"
            ))
        
        # Advanced grammar with spaCy (if available)
        if self.nlp and len(text) < 20000:
            try:
                doc = self.nlp(text[:20000])  # Limit for performance
                
                # Check for very short sentences (fragments)
                for sent in doc.sents:
                    words = [t for t in sent if not t.is_punct and not t.is_space]
                    if len(words) < 3 and len(sent.text) > 10:
                        errors.append(LinguisticError(
                            error_type=ErrorType.GRAMMAR,
                            text=sent.text[:50],
                            severity=0.4,
                            explanation="Possible sentence fragment"
                        ))
            except Exception as e:
                self.logger.debug(f"spaCy grammar check error: {e}")
        
        return errors
    
    def _check_style(self, text: str) -> List[LinguisticError]:
        """Check for style issues."""
        errors = []
        
        # Informal language
        for pattern in self.INFORMAL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                errors.append(LinguisticError(
                    error_type=ErrorType.STYLE,
                    text=match.group(),
                    severity=0.5,
                    explanation="Informal language in academic context"
                ))
        
        # Contractions
        for pattern in self.CONTRACTIONS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                errors.append(LinguisticError(
                    error_type=ErrorType.STYLE,
                    text=match.group(),
                    severity=0.4,
                    explanation="Avoid contractions in formal writing"
                ))
        
        # Check passive voice density
        passive_patterns = [
            r'\bis\s+\w+ed\b', r'\bwas\s+\w+ed\b', 
            r'\bwere\s+\w+ed\b', r'\bhas\s+been\s+\w+ed\b'
        ]
        passive_count = sum(len(re.findall(p, text)) for p in passive_patterns)
        sentences = len(re.split(r'[.!?]+', text))
        
        if sentences > 0 and passive_count / sentences > 0.7:
            errors.append(LinguisticError(
                error_type=ErrorType.STYLE,
                text="[document-wide]",
                severity=0.3,
                explanation="Very high use of passive voice"
            ))
        
        return errors
    
    def _check_terminology(self, text: str) -> List[LinguisticError]:
        """Check terminology consistency."""
        errors = []
        
        for canonical, variations in self.TERM_VARIATIONS.items():
            all_forms = variations | {canonical}
            found_forms = set()
            
            for form in all_forms:
                if re.search(r'\b' + re.escape(form) + r'\b', text, re.IGNORECASE):
                    found_forms.add(form)
            
            # If multiple forms used, flag inconsistency
            if len(found_forms) > 1:
                errors.append(LinguisticError(
                    error_type=ErrorType.TERMINOLOGY,
                    text=f"{canonical} variations",
                    severity=0.6,
                    suggestion=canonical,
                    explanation=f"Inconsistent use: {', '.join(found_forms)}"
                ))
        
        return errors
    
    def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics."""
        if not TEXTSTAT_AVAILABLE or not text:
            return {}
        
        try:
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text)
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
        
        # Small bonus for good readability
        if TEXTSTAT_AVAILABLE:
            try:
                reading_ease = textstat.flesch_reading_ease(text)
                if 20 <= reading_ease <= 60:  # Good for academic
                    score = min(1.0, score + 0.05)
            except Exception:
                pass
        
        return max(0.0, score)
    
    def _categorize_errors(self, errors: List[LinguisticError]) -> Dict[str, int]:
        """Count errors by type."""
        categories = defaultdict(int)
        for error in errors:
            categories[error.error_type.value] += 1
        return dict(categories)
    
    def _generate_feedback(self, errors: List[LinguisticError], text: str) -> str:
        """Generate human-readable feedback."""
        if not errors:
            return "Excellent linguistic quality with no significant issues detected."
        
        parts = []
        error_cats = self._categorize_errors(errors)
        
        # Spelling feedback
        if 'spelling' in error_cats:
            count = error_cats['spelling']
            parts.append(f"Found {count} spelling issue{'s' if count > 1 else ''}. "
                        "Thorough proofreading recommended.")
        
        # Grammar feedback
        if 'grammar' in error_cats:
            count = error_cats['grammar']
            parts.append(f"Detected {count} grammar/punctuation issue{'s' if count > 1 else ''}. "
                        "Review sentence structure.")
        
        # Style feedback
        if 'style' in error_cats:
            count = error_cats['style']
            parts.append(f"Found {count} style issue{'s' if count > 1 else ''}. "
                        "Maintain formal academic tone.")
        
        # Terminology feedback
        if 'terminology' in error_cats:
            count = error_cats['terminology']
            parts.append(f"Detected {count} terminology inconsistency{'ies' if count > 1 else ''}. "
                        "Use technical terms consistently.")
        
        # Readability feedback
        if TEXTSTAT_AVAILABLE and text:
            try:
                ease = textstat.flesch_reading_ease(text)
                if ease > 70:
                    parts.append("Text may be too simple for academic audience.")
                elif ease < 20:
                    parts.append("Text complexity is very high; consider simplifying.")
            except Exception:
                pass
        
        return " ".join(parts) if parts else "Minor issues detected. Overall quality is good."
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create result for empty papers."""
        return {
            "pillar_name": "Linguistic Quality",
            "score": 0.0,
            "feedback": "No text available for analysis.",
            "total_errors": 0,
            "error_breakdown": {},
            "readability": {}
        }
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create result when analysis fails."""
        return {
            "pillar_name": "Linguistic Quality",
            "score": 0.5,
            "feedback": f"Analysis encountered errors: {error_msg}. Partial results only.",
            "total_errors": 0,
            "error_breakdown": {},
            "readability": {}
        }


def evaluate(paper) -> dict:
    """
    Main evaluation function for compatibility with existing system.
    
    Args:
        paper: Paper object with text content
        
    Returns:
        dict: Score and feedback for linguistic quality
    """
    analyzer = LinguisticAnalyzer()
    return analyzer.analyze(paper)