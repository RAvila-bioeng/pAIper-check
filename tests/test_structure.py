import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from modules import check_structure
from models.paper import Paper

def test_structure_detects_missing_sections():
    text = "Abstract\nResults\nReferences"
    paper = Paper(raw_text=text)
    result = check_structure.evaluate(paper)
    assert result["score"] < 1
