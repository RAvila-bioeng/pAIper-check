# src/models/paper.py
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Section:
    """Represents a logical section of a scientific paper."""
    title: str
    content: str


@dataclass
class Reference:
    """Represents a single bibliographic reference."""
    text: str
    doi: Optional[str] = None
    url: Optional[str] = None


@dataclass
class Paper:
    raw_text: str
    title: str = ""
    abstract: str = ""
    full_text: str = ""
    sections: List[Section] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize derived fields from raw_text."""
        if not self.full_text and self.raw_text:
            self.full_text = self.raw_text

    def get_section_titles(self) -> List[str]:
        return [section.title for section in self.sections]

    def get_total_length(self) -> int:
        """Returns the total length of the manuscript in characters."""
        return len(self.full_text)
    
    def get_section_content(self, section_title: str) -> str:
        """Get content of a specific section by title."""
        for section in self.sections:
            if section.title.lower() == section_title.lower():
                return section.content
        return ""
