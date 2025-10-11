import re
from src.models.score import Score

def evaluate(paper):
    sections = ["abstract", "methodology", "results", "references"]
    missing = [s for s in sections if not re.search(s, paper.raw_text.lower())]
    completeness = 1 - len(missing)/len(sections)
    feedback = "Missing sections: " + ", ".join(missing) if missing else "All sections present."
    return Score("Structure & Completeness", completeness, feedback).__dict__
