# src/models/score.py
from dataclasses import dataclass

from typing import Dict, Any

@dataclass
class PillarResult:
    """Stores evaluation result for a single pillar (module)."""
    pillar_name: str
    score: float
    feedback: str
    gpt_analysis_data: Dict[str, Any] = None
