import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from models.paper import Paper
from modules import (
    check_structure, check_linguistics, check_cohesion, 
    check_reproducibility, check_references, check_quality
)
from utils.pdf_parser import parse_paper_from_pdf, extract_text_from_pdf
from config import config, setup_logging

# Setup logging
setup_logging(config)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run pAIper Check evaluation")
    parser.add_argument("--input", required=True, help="Path to input paper (PDF or TXT)")
    parser.add_argument("--output", help="Path to save evaluation report (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    try:
        # Parse the paper
        logger.info(f"Parsing paper: {args.input}")
        paper = parse_paper_from_pdf(str(input_path))
        
        if args.verbose:
            logger.info(f"Paper title: {paper.title}")
            logger.info(f"Number of sections: {len(paper.sections)}")
            logger.info(f"Number of references: {len(paper.references)}")

        # Run all evaluation modules
        logger.info("Running evaluation modules...")
        results = {}
        
        # Structure evaluation
        logger.info("Evaluating structure...")
        results["structure"] = check_structure.evaluate(paper)
        
        # Linguistics evaluation
        logger.info("Evaluating linguistics...")
        results["linguistics"] = check_linguistics.evaluate(paper)
        
        # Cohesion evaluation
        logger.info("Evaluating cohesion...")
        results["cohesion"] = check_cohesion.evaluate(paper)
        
        # Reproducibility evaluation
        logger.info("Evaluating reproducibility...")
        results["reproducibility"] = check_reproducibility.evaluate(paper)
        
        # References evaluation
        logger.info("Evaluating references...")
        results["references"] = check_references.evaluate(paper)
        
        # Quality evaluation
        logger.info("Evaluating scientific quality...")
        results["quality"] = check_quality.evaluate(paper)
        
        # Calculate overall score
        weights = config.get_evaluation_weights()
        overall_score = sum(results[pillar]["score"] * weights[pillar] for pillar in results.keys())
        
        # Prepare final results
        final_results = {
            "paper_info": {
                "title": paper.title,
                "abstract_length": len(paper.abstract),
                "total_sections": len(paper.sections),
                "total_references": len(paper.references),
                "total_length": paper.get_total_length()
            },
            "overall_score": overall_score,
            "pillar_scores": results,
            "weights": weights
        }
        
        # Print results
        print("\n" + "="*60)
        print("pAIper Check Evaluation Results")
        print("="*60)
        print(f"Paper: {paper.title or 'Untitled'}")
        print(f"Overall Score: {overall_score:.2f}/1.0")
        print("\nPillar Scores:")
        
        for pillar, result in results.items():
            score = result["score"]
            weight = weights[pillar]
            print(f"  - {result['pillar_name']}: {score:.2f} (weight: {weight:.2f})")
        
        print("\nFeedback Summary:")
        for pillar, result in results.items():
            print(f"  - {result['pillar_name']}: {result['feedback']}")
        
        # Save results if output path specified
        if args.output:
            import json
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {args.output}")
        
        print("\nEvaluation complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
