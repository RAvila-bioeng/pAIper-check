import argparse
from src.models.paper import Paper
from src.modules import check_structure, check_linguistics
from src.utils.pdf_parser import extract_text_from_pdf
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

from src.config import setup_logging
setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Run pAIper Check evaluation")
    parser.add_argument("--input", required=True, help="Path to input paper (PDF or TXT)")
    args = parser.parse_args()

    text = extract_text_from_pdf(args.input)
    paper = Paper(raw_text=text)

    results = {
        "structure": check_structure.evaluate(paper),
        "linguistics": check_linguistics.evaluate(paper),
    }

    print("\n✅ Evaluation complete:\n", results)

if __name__ == "__main__":
    main()
