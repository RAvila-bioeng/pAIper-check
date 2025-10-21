"""
Example Usage: pAIper-check with GPT-4o-mini Integration
Complete workflow from paper loading to GPT-enhanced analysis
"""

import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your modules (adjust paths as needed)
from modules import check_cohesion
from modules.openai_analyzer import GPTCoherenceAnalyzer

# Mock Paper class for demonstration
class Paper:
    """Mock paper object - replace with your actual Paper class"""
    def __init__(self, title, full_text, sections=None):
        self.title = title
        self.full_text = full_text
        self.sections = sections or []


def analyze_paper_complete(paper_path: str, use_gpt: bool = True) -> dict:
    """
    Complete analysis workflow for a paper.
    
    Args:
        paper_path: Path to the paper file (PDF or TXT)
        use_gpt: Whether to use GPT-4o-mini for deep analysis
    
    Returns:
        Complete analysis results
    """
    
    print(f"\n{'='*60}")
    print(f"ANALYZING PAPER: {os.path.basename(paper_path)}")
    print(f"{'='*60}\n")
    
    # STEP 1: Load paper (you would use your PDF parser here)
    print("📄 Step 1: Loading paper...")
    # For demo, we'll use mock data
    paper = Paper(
        title="Sample Scientific Paper",
        full_text=load_paper_text(paper_path),
        sections=[]  # You would extract sections here
    )
    print(f"   ✓ Loaded: {len(paper.full_text.split())} words\n")
    
    # STEP 2: Basic structural analysis (FREE)
    print("🔍 Step 2: Structural analysis (regex-based)...")
    basic_result = check_cohesion.evaluate(paper, use_gpt=False)
    print(f"   ✓ Basic Score: {basic_result['score']:.2f}/1.0")
    print(f"   ✓ Cost: $0.00 (FREE)\n")
    
    # STEP 3: GPT deep analysis (if enabled and needed)
    if use_gpt:
        print("🤖 Step 3: Deep AI analysis (GPT-4o-mini)...")
        
        # Check if GPT analysis is recommended
        if basic_result['score'] < 0.7:
            print("   ⚠️  Low score detected - GPT analysis recommended")
        
        enhanced_result = check_cohesion.evaluate(paper, use_gpt=True)
        
        gpt_info = enhanced_result.get('gpt_analysis', {})
        if gpt_info.get('used', False):
            cost = gpt_info.get('cost_info', {}).get('cost_usd', 0)
            print(f"   ✓ Enhanced Score: {enhanced_result['score']:.2f}/1.0")
            print(f"   ✓ Cost: ${cost:.4f}")
            
            # Show cost report if available
            cost_report = enhanced_result.get('cost_report', {})
            if cost_report:
                print(f"   ✓ Average cost/paper: ${cost_report.get('average_cost_per_paper', 0):.4f}")
        else:
            print(f"   ℹ️  GPT analysis skipped: {gpt_info.get('reason', 'Unknown')}")
            enhanced_result = basic_result
    else:
        print("🤖 Step 3: Skipped (GPT disabled)")
        enhanced_result = basic_result
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")
    
    return enhanced_result


def load_paper_text(paper_path: str) -> str:
    """
    Load paper text from file.
    Replace this with your actual PDF parser.
    """
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {paper_path}")
        return ""
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return ""


def display_analysis_results(result: dict):
    """Display analysis results in a formatted way."""
    
    print("\n" + "="*60)
    print("📊 COHERENCE & COHESION ANALYSIS RESULTS")
    print("="*60 + "\n")
    
    # Overall score
    score = result.get('score', 0)
    print(f"Overall Score: {score:.2f}/1.0 ", end="")
    
    if score >= 0.8:
        print("✅ EXCELLENT")
    elif score >= 0.7:
        print("✓ GOOD")
    elif score >= 0.5:
        print("⚠️  FAIR")
    else:
        print("❌ NEEDS IMPROVEMENT")
    
    print("\n" + "-"*60)
    
    # Feedback
    feedback = result.get('feedback', 'No feedback available')
    print("\n📝 FEEDBACK:\n")
    print(feedback)
    
    # GPT Analysis info
    gpt_analysis = result.get('gpt_analysis', {})
    if gpt_analysis.get('success'):
        print("\n" + "-"*60)
        print("🤖 GPT-4o-mini DEEP ANALYSIS")
        print("-"*60)
        
        analysis_data = gpt_analysis.get('analysis', {})
        
        # Detailed scores
        if 'semantic_coherence' in analysis_data:
            sem_score = analysis_data['semantic_coherence'].get('score', 0)
            print(f"\n  Semantic Coherence: {sem_score:.2f}/1.0")
        
        if 'logical_flow' in analysis_data:
            flow_score = analysis_data['logical_flow'].get('score', 0)
            print(f"  Logical Flow: {flow_score:.2f}/1.0")
        
        # Issues
        issues = analysis_data.get('issues', [])
        if issues:
            print(f"\n  🔴 Issues Found: {len(issues)}")
            for i, issue in enumerate(issues[:3], 1):
                severity = issue.get('severity', 'unknown')
                issue_type = issue.get('type', 'unknown')
                description = issue.get('description', 'N/A')
                print(f"     {i}. [{severity.upper()}] {issue_type}: {description[:100]}...")
        
        # Suggestions
        suggestions = analysis_data.get('suggestions', [])
        if suggestions:
            print(f"\n  💡 Recommendations: {len(suggestions)}")
            for i, sugg in enumerate(suggestions[:3], 1):
                priority = sugg.get('priority', 'medium')
                action = sugg.get('action', 'N/A')
                print(f"     {i}. [{priority.upper()}] {action[:100]}...")
    
    # Cost report
    cost_report = result.get('cost_report', {})
    if cost_report and cost_report.get('total_papers_analyzed', 0) > 0:
        print("\n" + "-"*60)
        print("💰 COST REPORT")
        print("-"*60)
        print(f"  Papers Analyzed: {cost_report.get('total_papers_analyzed', 0)}")
        print(f"  Total Cost: ${cost_report.get('total_cost_usd', 0):.4f}")
        print(f"  Avg Cost/Paper: ${cost_report.get('average_cost_per_paper', 0):.4f}")
        
        projections = cost_report.get('cost_projections', {})
        if projections:
            print(f"\n  Cost Projections:")
            print(f"    100 papers: ${projections.get('100_papers', 0):.2f}")
            print(f"    500 papers: ${projections.get('500_papers', 0):.2f}")
            print(f"    1000 papers: ${projections.get('1000_papers', 0):.2f}")
    
    print("\n" + "="*60 + "\n")


def batch_analyze_papers(paper_paths: list, use_gpt: bool = True) -> list:
    """
    Analyze multiple papers in batch.
    
    Args:
        paper_paths: List of paths to paper files
        use_gpt: Whether to use GPT analysis
    
    Returns:
        List of analysis results
    """
    
    results = []
    total_cost = 0.0
    
    print(f"\n{'='*60}")
    print(f"BATCH ANALYSIS: {len(paper_paths)} papers")
    print(f"{'='*60}\n")
    
    for i, paper_path in enumerate(paper_paths, 1):
        print(f"\n[{i}/{len(paper_paths)}] Processing: {os.path.basename(paper_path)}")
        
        result = analyze_paper_complete(paper_path, use_gpt)
        results.append(result)
        
        # Track costs
        if 'cost_report' in result:
            cost_report = result['cost_report']
            paper_cost = cost_report.get('total_cost_usd', 0)
            total_cost += paper_cost
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Papers processed: {len(results)}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Average cost per paper: ${total_cost/max(1, len(results)):.4f}")
    print(f"{'='*60}\n")
    
    return results


def save_results_to_json(results: dict, output_path: str):
    """Save analysis results to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def compare_basic_vs_gpt_analysis(paper_path: str):
    """
    Compare basic analysis vs GPT-enhanced analysis.
    Useful for evaluating the value of GPT analysis.
    """
    
    print(f"\n{'='*60}")
    print("COMPARISON: Basic vs GPT-Enhanced Analysis")
    print(f"{'='*60}\n")
    
    # Load paper
    paper = Paper(
        title="Comparison Test Paper",
        full_text=load_paper_text(paper_path),
        sections=[]
    )
    
    # Basic analysis
    print("1️⃣  Running BASIC analysis (FREE)...")
    basic_result = check_cohesion.evaluate(paper, use_gpt=False)
    basic_score = basic_result['score']
    print(f"   Score: {basic_score:.2f}/1.0")
    print(f"   Cost: $0.00")
    
    # GPT-enhanced analysis
    print("\n2️⃣  Running GPT-ENHANCED analysis...")
    gpt_result = check_cohesion.evaluate(paper, use_gpt=True)
    gpt_score = gpt_result['score']
    gpt_cost = gpt_result.get('cost_report', {}).get('total_cost_usd', 0)
    print(f"   Score: {gpt_score:.2f}/1.0")
    print(f"   Cost: ${gpt_cost:.4f}")
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Score Difference: {abs(gpt_score - basic_score):.2f} points")
    print(f"Score Change: {((gpt_score - basic_score) / basic_score * 100):.1f}%")
    print(f"Additional Cost: ${gpt_cost:.4f}")
    
    if abs(gpt_score - basic_score) > 0.1:
        print("\n✅ GPT analysis provided significant additional insights")
    else:
        print("\n✓ Basic analysis was sufficient for this paper")
    
    print(f"{'='*60}\n")
    
    return {
        'basic': basic_result,
        'gpt': gpt_result,
        'comparison': {
            'score_diff': gpt_score - basic_score,
            'cost': gpt_cost,
            'value': 'high' if abs(gpt_score - basic_score) > 0.1 else 'low'
        }
    }


# ============== MAIN EXECUTION EXAMPLES ==============

if __name__ == "__main__":
    """
    Run examples - choose which one to execute
    """
    
    print("\n" + "="*60)
    print("pAIper-check with GPT-4o-mini Integration")
    print("="*60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found!")
        print("\nPlease create a .env file with:")
        print('OPENAI_API_KEY="sk-your-actual-api-key-here"')
        print("\nOr set it as environment variable:")
        print("export OPENAI_API_KEY='sk-your-actual-api-key-here'")
        exit(1)
    
    print(f"\n✓ API Key found: {api_key[:20]}...")
    
    # ============== EXAMPLE 1: Single Paper Analysis ==============
    print("\n" + "="*60)
    print("EXAMPLE 1: Analyze Single Paper")
    print("="*60)
    
    # Replace with your actual paper path
    paper_path = "path/to/your/paper.txt"
    
    if os.path.exists(paper_path):
        result = analyze_paper_complete(paper_path, use_gpt=True)
        display_analysis_results(result)
        
        # Save results
        save_results_to_json(result, "analysis_result.json")
    else:
        print(f"\n⚠️  Paper file not found: {paper_path}")
        print("   Please update the paper_path variable with a valid file path")
    
    
    # ============== EXAMPLE 2: Batch Analysis ==============
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Analysis (Multiple Papers)")
    print("="*60)
    
    # Replace with your actual paper paths
    paper_list = [
        "paper1.txt",
        "paper2.txt",
        "paper3.txt"
    ]
    
    # Filter existing files
    existing_papers = [p for p in paper_list if os.path.exists(p)]
    
    if existing_papers:
        batch_results = batch_analyze_papers(existing_papers, use_gpt=True)
        
        # Save batch results
        save_results_to_json(
            {'papers': batch_results, 'count': len(batch_results)},
            "batch_analysis_results.json"
        )
    else:
        print("\n⚠️  No paper files found in paper_list")
        print("   Please add valid paper paths to paper_list")
    
    
    # ============== EXAMPLE 3: Comparison Analysis ==============
    print("\n" + "="*60)
    print("EXAMPLE 3: Compare Basic vs GPT Analysis")
    print("="*60)
    
    if os.path.exists(paper_path):
        comparison = compare_basic_vs_gpt_analysis(paper_path)
        
        # Save comparison
        save_results_to_json(comparison, "comparison_analysis.json")
    else:
        print(f"\n⚠️  Paper file not found for comparison")
    
    
    # ============== EXAMPLE 4: Cost Estimation ==============
    print("\n" + "="*60)
    print("EXAMPLE 4: Cost Estimation for Your Project")
    print("="*60)
    
    # IV Congreso SAM estimate
    expected_papers = 50
    avg_cost_per_paper = 0.003  # Conservative estimate
    
    print(f"\nFor IV Congreso SAM ({expected_papers} papers):")
    print(f"  Estimated total cost: ${expected_papers * avg_cost_per_paper:.2f}")
    print(f"  Cost per paper: ${avg_cost_per_paper:.3f}")
    
    print(f"\nCost breakdown:")
    print(f"  - Papers with score ≥ 0.7 (FREE): ~{int(expected_papers * 0.6)} papers = $0.00")
    print(f"  - Papers with score < 0.7 (GPT): ~{int(expected_papers * 0.4)} papers = ${int(expected_papers * 0.4) * avg_cost_per_paper:.2f}")
    
    print(f"\nGeneral cost projections:")
    print(f"  100 papers: ${100 * avg_cost_per_paper:.2f}")
    print(f"  500 papers: ${500 * avg_cost_per_paper:.2f}")
    print(f"  1000 papers: ${1000 * avg_cost_per_paper:.2f}")
    
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)
    print("\n💡 TIP: Modify paper paths and settings to test with your actual papers")
    print("💡 TIP: Check the generated JSON files for detailed results")
    print("💡 TIP: Set use_gpt=False to test without GPT (FREE)")
    print("\n")