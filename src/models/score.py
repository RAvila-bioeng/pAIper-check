# src/models/score.py
from dataclasses import dataclass

@dataclass
class PillarResult:
    """Stores evaluation result for a single pillar (module)."""
    pillar_name: str
    score: float
    feedback: str
