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