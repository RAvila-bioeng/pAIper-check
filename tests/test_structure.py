from src.modules import check_structure
from src.models.models import Paper

def test_structure_detects_missing_sections():
    text = "Abstract\nResults\nReferences"
    paper = Paper(raw_text=text)
    result = check_structure.evaluate(paper)
    assert result["value"] < 1
