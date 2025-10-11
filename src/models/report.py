# src/models/report.py
from dataclasses import dataclass, field
from typing import List
from .paper import Paper
from .score import PillarResult


@dataclass
class Report:
    """Aggregated evaluation report."""
    paper: Paper
    pillar_results: List[PillarResult] = field(default_factory=list)
    final_score: float = 0.0

    def compute_final_score(self) -> float:
        """Compute the average score across all pillars."""
        if not self.pillar_results:
            self.final_score = 0.0
        else:
            self.final_score = sum(p.score for p in self.pillar_results) / len(self.pillar_results)
        return self.final_score

    def to_dict(self) -> dict:
        """Serialize the report for JSON export."""
        return {
            "paper_title": self.paper.title,
            "final_score": self.final_score,
            "pillar_results": [
                {"pillar_name": p.pillar_name, "score": p.score, "feedback": p.feedback}
                for p in self.pillar_results
            ],
        }
