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


def test_structure_detects_out_of_order_sections():
    text = (
        "Introduction\n...\n"
        "Conclusion\n...\n"
        "Methods\n...\n"
        "Results\n...\n"
        "References\n...\n"
    )
    paper = Paper(raw_text=text)
    result = check_structure.evaluate(paper)
    assert result["score"] < 1


def test_structure_short_sections_penalized():
    paper = Paper(
        raw_text=(
            "Abstract\nshort\n"
            "Introduction\nshort\n"
            "Methods\nshort\n"
            "Results\nshort\n"
            "Discussion\nshort\n"
            "Conclusion\nshort\n"
            "References\nR1\n"
        ),
        sections=[
            # Very short contents to trigger penalty
            
        ],
    )
    # Provide sections metadata with short content
    from models.paper import Section
    paper.sections = [
        Section(title="Abstract", content="short"),
        Section(title="Introduction", content="tiny"),
        Section(title="Methods", content="tiny"),
        Section(title="Results", content="tiny"),
        Section(title="Discussion", content="tiny"),
        Section(title="Conclusion", content="tiny"),
        Section(title="References", content="R1"),
    ]

    result = check_structure.evaluate(paper)
    assert result["score"] < 1