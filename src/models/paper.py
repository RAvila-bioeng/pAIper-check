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
    """Main model representing a scientific paper."""
    title: str
    abstract: str
    full_text: str
    sections: List[Section] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)

    def get_section_titles(self) -> List[str]:
        return [section.title for section in self.sections]

    def get_total_length(self) -> int:
        """Returns the total length of the manuscript in characters."""
        return len(self.full_text)
