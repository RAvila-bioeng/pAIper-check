import argparse
import logging
import sys
import json
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
    parser = argparse.ArgumentParser(
        description="Run pAIper Check evaluation with optional GPT-4o-mini deep analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation (FREE)
  python main.py --input paper.pdf
  
  # With LLM deep analysis (recommended)
  python main.py --input paper.pdf --use-llm
  
  # Show detailed linguistic errors
  python main.py --input paper.pdf --show-errors
  
  # Save detailed results to JSON
  python main.py --input paper.pdf --use-llm --output results.json
  
  # Batch analysis
  python main.py --input papers/*.pdf --use-llm --batch
        """
    )
    
    parser.add_argument("--input", required=True, help="Path to input paper (PDF or TXT)")
    parser.add_argument("--output", help="Path to save evaluation report (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--use-llm", action="store_true", 
                       help="Enable deep analysis with GPT-4o-mini and Perplexity Sonar")
    parser.add_argument("--force-gpt", action="store_true",
                       help="Force GPT analysis even if basic score is good")
    parser.add_argument("--gpt-report", action="store_true",
                       help="Show detailed GPT cost report at the end")
    parser.add_argument("--batch", action="store_true",
                       help="Process multiple papers (use with wildcards)")
    
    # NEW: Linguistic error reporting options
    parser.add_argument("--show-errors", action="store_true",
                       help="Show detailed linguistic errors with locations")
    parser.add_argument("--max-errors", type=int, default=30,
                       help="Maximum number of errors to display (default: 30)")
    parser.add_argument("--export-errors", 
                       help="Export linguistic errors to file (JSON, CSV, or HTML)")
    
    args = parser.parse_args()

    # Check if GPT is enabled and API key is available
    if args.use_llm:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("GPT analysis requested but OPENAI_API_KEY not found in .env")
            logger.warning("Add to .env file: OPENAI_API_KEY=sk-your-key-here")
            logger.warning("Continuing with basic analysis only...")
            args.use_llm = False
        else:
            logger.info("✓ GPT-4o-mini enabled for deep coherence analysis")

    # Validate input file
    input_path = Path(args.input)
    
    # Handle batch mode with wildcards
    if args.batch or '*' in args.input:
        import glob
        paper_files = glob.glob(args.input)
        
        if not paper_files:
            logger.error(f"No files found matching: {args.input}")
            return 1
        
        logger.info(f"Batch mode: Found {len(paper_files)} papers")
        return process_batch(paper_files, args)
    
    # Single paper mode
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    try:
        result = process_single_paper(str(input_path), args)
        
        if result is None:
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def log_llm_status(result: dict, pillar_name: str):
    """Helper to log the status of an LLM analysis."""
    gpt_info = result.get("gpt_analysis", {})
    if not gpt_info.get("used"):
        reason = gpt_info.get("reason", "Unknown")
        logger.info(f"  → {pillar_name} LLM analysis skipped: {reason}")
        return

    if gpt_info.get("success"):
        cost = gpt_info.get("cost_info", {}).get("cost_usd", 0)
        engine = gpt_info.get("model", "LLM")
        logger.info(f"  → {pillar_name} analysis with {engine} successful (cost: ${cost:.4f})")
    else:
        error = gpt_info.get("error", "Unknown error")
        logger.warning(f"  → {pillar_name} LLM analysis failed: {error}")

def process_single_paper(input_path, args):
    """Process a single paper"""
    
    # Parse the paper
    logger.info(f"Parsing paper: {input_path}")
    paper = parse_paper_from_pdf(input_path)
    
    if args.verbose:
        logger.info(f"Paper title: {paper.title}")
        logger.info(f"Number of sections: {len(paper.sections)}")
        logger.info(f"Number of references: {len(paper.references)}")

    # Run all evaluation modules
    logger.info("Running evaluation modules...")
    results = {}
    
    # Structure evaluation
    logger.info("Evaluating structure...")
    results["structure"] = check_structure.evaluate(paper, use_gpt=args.use_llm)
    if args.use_llm:
        log_llm_status(results["structure"], "Structure")
    
    # Linguistics evaluation
    logger.info("Evaluating linguistics...")
    results["linguistics"] = check_linguistics.evaluate(paper, use_gpt=args.use_llm)
    if args.use_llm:
        log_llm_status(results["linguistics"], "Linguistics")
    
    # Cohesion evaluation (with optional GPT)
    logger.info("Evaluating cohesion...")
    results["cohesion"] = check_cohesion.evaluate(paper, use_gpt=args.use_llm)
    if args.use_llm:
        log_llm_status(results["cohesion"], "Cohesion")

    # Reproducibility evaluation
    logger.info("Evaluating reproducibility...")
    results["reproducibility"] = check_reproducibility.evaluate(paper, use_gpt=args.use_llm)
    if args.use_llm:
        log_llm_status(results["reproducibility"], "Reproducibility")
    
    # References evaluation
    logger.info("Evaluating references...")
    results["references"] = check_references.evaluate(paper, use_gpt=args.use_llm)
    if args.use_llm:
        log_llm_status(results["references"], "References")
    
    # Quality evaluation (with optional GPT)
    logger.info("Evaluating scientific quality...")
    results["quality"] = check_quality.evaluate(paper, use_gpt=args.use_llm)
    if args.use_llm:
        log_llm_status(results["quality"], "Quality")
    
    # Calculate overall score
    weights = config.get_evaluation_weights()
    overall_score = sum(results[pillar]["score"] * weights[pillar] for pillar in results.keys())
    
    # Prepare final results
    final_results = {
        "paper_info": {
            "title": paper.title,
            "file_path": input_path,
            "abstract_length": len(paper.abstract) if paper.abstract else 0,
            "total_sections": len(paper.sections),
            "total_references": len(paper.references),
            "total_length": paper.get_total_length() if hasattr(paper, 'get_total_length') else 0
        },
        "overall_score": overall_score,
        "pillar_scores": results,
        "weights": weights,
        "gpt_enabled": args.use_llm
    }
    
    # Add GPT cost report if available
    if args.use_llm and "cohesion" in results:
        cost_report = results["cohesion"].get("cost_report")
        if cost_report:
            final_results["gpt_cost_report"] = cost_report
    
    # Print results
    print_results(paper, overall_score, results, weights, args)
    
    
    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {args.output}")
    
    return final_results


def process_batch(paper_files, args):
    """Process multiple papers in batch"""
    
    print(f"\n{'='*70}")
    print(f"📚 BATCH ANALYSIS: {len(paper_files)} papers")
    print(f"{'='*70}\n")
    
    all_results = []
    total_cost = 0.0
    successful = 0
    failed = 0
    
    for i, paper_file in enumerate(paper_files, 1):
        print(f"\n[{i}/{len(paper_files)}] Processing: {Path(paper_file).name}")
        print("-" * 70)
        
        try:
            # Create args copy for this paper
            paper_args = argparse.Namespace(**vars(args))
            paper_args.input = paper_file
            
            # Process paper
            result = process_single_paper(paper_file, paper_args)
            
            if result:
                all_results.append(result)
                successful += 1
                
                # Track GPT costs
                if "gpt_cost_report" in result:
                    paper_cost = result["gpt_cost_report"].get("total_cost_usd", 0)
                    total_cost += paper_cost
            else:
                failed += 1
                
        except Exception as e:
            logger.error(f"Error processing {paper_file}: {e}")
            failed += 1
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Print batch summary
    print_batch_summary(all_results, successful, failed, total_cost, args)
    
    # Save batch results
    if args.output:
        batch_output = {
            "batch_info": {
                "total_papers": len(paper_files),
                "successful": successful,
                "failed": failed,
                "gpt_enabled": args.use_llm,  # ✅ CORRECCIÓN: era use_chatgpt
                "total_gpt_cost": total_cost
            },
            "papers": all_results
        }
        
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(batch_output, f, indent=2, ensure_ascii=False)
        logger.info(f"Batch results saved to: {args.output}")
    
    return 0 if failed == 0 else 1


def print_linguistic_errors(ling_result, max_errors=30):
    """Print detailed linguistic errors with locations"""
    
    errors = ling_result.get('error_details', [])
    if not errors:
        print("\n  ✅ No linguistic errors detected!")
        return
    
    total_errors = len(errors)
    breakdown = ling_result.get('error_breakdown', {})
    
    print(f"\n  📋 Linguistic Error Details ({total_errors} total errors):")
    print(f"  {'─'*66}")
    
    # Show breakdown
    if breakdown:
        print("\n  Error Types:")
        for error_type, count in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
            emoji = {
                'spelling': '📝',
                'grammar': '📖',
                'punctuation': '❗',
                'style': '✨',
                'terminology': '🔤'
            }.get(error_type, '•')
            print(f"    {emoji} {error_type.capitalize()}: {count}")
    
    # Show first N errors
    print(f"\n  Showing first {min(max_errors, total_errors)} errors:")
    print(f"  {'─'*66}")
    
    # Group by type for better readability
    errors_by_type = {}
    for error in errors[:max_errors]:
        error_type = error['type']
        if error_type not in errors_by_type:
            errors_by_type[error_type] = []
        errors_by_type[error_type].append(error)
    
    # Print grouped errors
    for error_type, type_errors in sorted(errors_by_type.items()):
        emoji = {
            'spelling': '📝',
            'grammar': '📖',
            'punctuation': '❗',
            'style': '✨',
            'terminology': '🔤'
        }.get(error_type, '•')
        
        print(f"\n  {emoji} {error_type.upper()} ({len(type_errors)} shown)")
        
        for i, error in enumerate(type_errors[:10], 1):  # Limit per type to 10
            severity_emoji = '🔴' if error['severity'] >= 0.7 else '🟡' if error['severity'] >= 0.4 else '🟢'
            
            print(f"\n    {i}. {severity_emoji} Line {error['line']}")
            print(f"       Found: '{error['text']}'")
            
            if error.get('suggestion'):
                print(f"       → Suggest: '{error['suggestion']}'")
            
            print(f"       Context: {error['context']}")
            print(f"       Reason: {error['explanation']}")
    
    if total_errors > max_errors:
        remaining = total_errors - max_errors
        print(f"\n  ... and {remaining} more errors")
        print(f"  💡 Use --max-errors {total_errors} to see all errors")


def print_gpt_analysis_section(title: str, gpt_info: dict):
    """
    Helper to print a generic GPT analysis section.
    Handles both OpenAI and Perplexity outputs.
    """
    if not gpt_info:
        return
        
    # Determinar el motor de análisis (OpenAI o Perplexity)
    engine_name = gpt_info.get('model', 'Unknown LLM')
    
    # ✅ CORRECCIÓN: Definir las variables de detección ANTES de usarlas
    is_openai = 'gpt' in engine_name.lower()
    is_perplexity = 'perplexity' in engine_name.lower() or 'sonar' in engine_name.lower()
    
    # Ajustar nombre del motor para display
    if is_openai:
        engine_name = "GPT-4o-mini"
    elif is_perplexity:
        engine_name = "Perplexity Sonar Pro"
    
    print("\n" + "-"*70)
    print(f"🤖 {engine_name} Deep Analysis: {title}")
    print("-"*70)

    if not gpt_info.get("success"):
        error = gpt_info.get('error', 'Unknown error')
        print(f"  Analysis Failed: {error}")
        return
    
    # --- Detalles específicos de OpenAI ---
    if is_openai:
        cost_info = gpt_info.get("cost_info", {})
        print(f"  Cost: ${cost_info.get('cost_usd', 0):.4f}")
        print(f"  Tokens: {cost_info.get('total_tokens', 0)} (input: {cost_info.get('input_tokens', 0)}, output: {cost_info.get('output_tokens', 0)})")
    
    # --- Contenido del análisis (común a ambos) ---
    analysis = gpt_info.get("analysis", {})
    
    # Si el análisis es un string (de Perplexity), simplemente lo imprimimos
    if isinstance(analysis, str):
        print(analysis)
        return

    # Si es un diccionario (de OpenAI), lo formateamos
    if isinstance(analysis, dict):
        if 'overall_score' in analysis:
            print(f"  Overall GPT Score: {analysis.get('overall_score', 0):.2f} - Verdict: {analysis.get('final_verdict', 'N/A')}")
        
        if 'sub_modules' in analysis:
            print(f"\n  {title} Sub-modules:")
            for name, sub in analysis['sub_modules'].items():
                if sub:
                    print(f"    - {name.replace('_', ' ').title():.<25} {sub.get('score', 0):.2f} - {sub.get('feedback', 'N/A')}")

        strengths = analysis.get('strengths', [])
        if strengths:
            print("\n  Strengths:")
            for strength in strengths:
                print(f"    - {strength}")

        issues = analysis.get('issues', [])
        if issues:
            print("\n  Issues Found:")
            for issue in issues:
                print(f"    - {issue}")
        
        suggestions = analysis.get('suggestions', [])
        if suggestions:
            print("\n  Recommendations:")
            for suggestion in suggestions:
                print(f"    - {suggestion}")


def print_results(paper, overall_score, results, weights, args):
    """Print formatted results"""
    
    print("\n" + "="*70)
    print("📋 pAIper Check Evaluation Results")
    print("="*70)
    print(f"Paper: {paper.title or 'Untitled'}")
    print(f"Overall Score: {overall_score:.2f}/1.0 ", end="")
    
    # Add emoji based on score
    if overall_score >= 0.8:
        print("🟢 EXCELLENT")
    elif overall_score >= 0.7:
        print("🟡 GOOD")
    elif overall_score >= 0.5:
        print("🟠 FAIR")
    else:
        print("🔴 NEEDS IMPROVEMENT")
    
    print("\n" + "-"*70)
    print("Pillar Scores:")
    print("-"*70)
    
    for pillar, result in results.items():
        score = result["score"]
        weight = weights[pillar]
        pillar_name = result.get('pillar_name', pillar.title())
        
        # Score indicator
        if score >= 0.8:
            indicator = "🟢"
        elif score >= 0.7:
            indicator = "🟡"
        elif score >= 0.5:
            indicator = "🟠"
        else:
            indicator = "🔴"
        
        print(f"  {indicator} {pillar_name:.<40} {score:.2f} (weight: {weight:.2f})")
        
        # Print score breakdown if available
        if 'score_breakdown' in result:
            for sub_name, sub_score in result['score_breakdown'].items():
                print(f"    - {sub_name:.<37} {sub_score:.2f}")

    print("\n" + "-"*70)
    print("Detailed Feedback:")
    print("-"*70)
    
    for pillar, result in results.items():
        pillar_name = result.get('pillar_name', pillar.title())
        feedback = result.get('feedback', 'No feedback available')
        
        print(f"\n{pillar_name}:")
        print(f"  {feedback}")
        
        # Show linguistic errors if requested or if score is very low
        if pillar == "linguistics" and (args.show_errors or result.get("score", 1.0) < 0.3):
            print_linguistic_errors(result, args.max_errors)
    
    # Show GPT/LLM analysis if used
    if args.use_llm:
        # Definición correcta de pilares con sus claves exactas
        # IMPORTANTE: Las claves deben coincidir con las del diccionario results
        llm_pillars = [
            ("Structure & Completeness", "structure"),
            ("Linguistic Quality", "linguistics"),
            ("Cohesion & Flow", "cohesion"),
            ("Reproducibility", "reproducibility"),
            ("References & Citations", "references"),
            ("Scientific Quality", "quality")
        ]

        for pillar_name, pillar_key in llm_pillars:
            # Verificar que el pilar existe en los resultados
            if pillar_key not in results:
                continue
            
            # Obtener información del análisis GPT/LLM
            result = results[pillar_key]
            gpt_info = result.get("gpt_analysis")
            
            # Verificar que el análisis fue intentado
            if gpt_info and gpt_info.get("used"):
                print_gpt_analysis_section(pillar_name, gpt_info)
    
    # GPT cost report
    if args.gpt_report and "cohesion" in results:
        cost_report = results["cohesion"].get("cost_report")
        if cost_report:
            print("\n" + "-"*70)
            print("💰 GPT Cost Report:")
            print("-"*70)
            print(f"  Papers analyzed with GPT: {cost_report.get('total_papers_analyzed', 0)}")
            print(f"  Total cost: ${cost_report.get('total_cost_usd', 0):.4f}")
            print(f"  Average cost/paper: ${cost_report.get('average_cost_per_paper', 0):.4f}")
            
            projections = cost_report.get('cost_projections', {})
            if projections:
                print(f"  Projected cost for 100 papers: ${projections.get('100_papers', 0):.2f}")
    
    print("\n" + "="*70)
    print("✓ Evaluation complete!")
    
    # Hint about error details
    if "linguistics" in results and results["linguistics"].get("total_errors", 0) > 0:
        if not args.show_errors:
            print("\n💡 Tip: Use --show-errors to see detailed linguistic error locations")
    
    print("="*70 + "\n")

    
     
def print_batch_summary(results, successful, failed, total_cost, args):
    """Print batch processing summary"""
    
    print("\n" + "="*70)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*70)
    
    total = successful + failed
    print(f"Total papers: {total}")
    print(f"  ✓ Successful: {successful}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}")
    
    if results:
        # Calculate average score
        avg_score = sum(r["overall_score"] for r in results) / len(results)
        print(f"\nAverage overall score: {avg_score:.2f}/1.0")
        
        # Score distribution
        excellent = sum(1 for r in results if r["overall_score"] >= 0.8)
        good = sum(1 for r in results if 0.7 <= r["overall_score"] < 0.8)
        fair = sum(1 for r in results if 0.5 <= r["overall_score"] < 0.7)
        poor = sum(1 for r in results if r["overall_score"] < 0.5)
        
        print(f"\nScore distribution:")
        print(f"  🟢 Excellent (≥0.8): {excellent} papers")
        print(f"  🟡 Good (0.7-0.8): {good} papers")
        print(f"  🟠 Fair (0.5-0.7): {fair} papers")
        print(f"  🔴 Poor (<0.5): {poor} papers")
    
    if args.use_llm and total_cost > 0:
        print(f"\n💰 GPT Analysis Costs:")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Average cost/paper: ${total_cost/max(1, successful):.4f}")
        
        # Papers that used GPT
        gpt_used = sum(1 for r in results if r.get("gpt_cost_report", {}).get("total_papers_analyzed", 0) > 0)
        print(f"  Papers analyzed with GPT: {gpt_used}/{successful}")
        print(f"  Cost savings: ~{((successful - gpt_used) * 0.002):.4f} (intelligent filtering)")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    exit(main())