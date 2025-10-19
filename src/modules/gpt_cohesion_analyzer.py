"""
OpenAI GPT-4o-mini Integration Module
Deep coherence analysis using AI for pAIper-check
Cost-optimized implementation: ~$0.002-0.004 per paper
"""

import os
import json
import time
from typing import Dict, Optional, List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cost tracking constants (per million tokens)
COST_PER_1M_INPUT = 0.15  # $0.15 per 1M input tokens
COST_PER_1M_OUTPUT = 0.60  # $0.60 per 1M output tokens


class GPTCoherenceAnalyzer:
    """
    Manages GPT-4o-mini analysis for deep coherence evaluation.
    Optimized for cost and quality.
    """
    
    def __init__(self):
        self.total_cost = 0.0
        self.total_papers = 0
        self.total_tokens_input = 0
        self.total_tokens_output = 0
        self.analysis_history = []
    
    def should_use_gpt_analysis(self, basic_score: float, feedback: str) -> bool:
        """
        Decide if GPT analysis is needed based on basic analysis results.
        Strategy: Only use GPT when basic score < 0.7 or critical issues detected.
        """
        # Always use GPT if score is low
        if basic_score < 0.7:
            return True
        
        # Check for critical keywords in feedback
        critical_keywords = [
            'inconsistent', 'poor', 'violation', 'missing',
            'insufficient', 'weak', 'limited', 'few'
        ]
        
        feedback_lower = feedback.lower()
        if any(keyword in feedback_lower for keyword in critical_keywords):
            return True
        
        return False
    
    def analyze_coherence(self, paper, gpt_analysis_data: Dict, 
                         basic_score: float) -> Dict:
        """
        Perform deep coherence analysis using GPT-4o-mini.
        
        Args:
            paper: Paper object
            gpt_analysis_data: Pre-prepared analysis data from check_cohesion
            basic_score: Score from basic structural analysis
        
        Returns:
            Dict with GPT analysis results and cost information
        """
        
        try:
            # Generate optimized prompt
            prompt = self._build_optimized_prompt(gpt_analysis_data, basic_score)
            
            # Call GPT-4o-mini
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=1500,   # Limit output to control costs
                response_format={"type": "json_object"}  # Get structured JSON
            )
            
            # Extract response
            analysis_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                analysis_json = json.loads(analysis_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                analysis_json = {
                    "overall_score": basic_score,
                    "analysis": analysis_text,
                    "issues": [],
                    "suggestions": []
                }
            
            # Calculate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            # Update tracking
            self.total_cost += cost
            self.total_papers += 1
            self.total_tokens_input += input_tokens
            self.total_tokens_output += output_tokens
            
            # Store analysis history
            self.analysis_history.append({
                'paper_title': gpt_analysis_data.get('paper_info', {}).get('title', 'Unknown'),
                'basic_score': basic_score,
                'gpt_score': analysis_json.get('overall_score', basic_score),
                'cost': cost,
                'tokens': {'input': input_tokens, 'output': output_tokens},
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return {
                'success': True,
                'analysis': analysis_json,
                'raw_response': analysis_text,
                'cost_info': {
                    'cost_usd': round(cost, 4),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens
                },
                'model': 'gpt-4o-mini'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'cost_info': {'cost_usd': 0.0}
            }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for GPT-4o-mini."""
        return """You are an expert scientific paper reviewer specializing in coherence, cohesion, and logical flow analysis.

Your task is to provide a deep, semantic analysis of academic papers, focusing on:
- Logical connection between ideas
- Argumentative structure and flow
- Semantic coherence and consistency
- Quality of transitions and connections

Provide specific, actionable feedback that goes beyond structural analysis.
Focus on the MEANING and LOGIC, not just the presence of connector words.

IMPORTANT: Always respond with valid JSON format."""
    
    def _build_optimized_prompt(self, gpt_analysis_data: Dict, 
                               basic_score: float) -> str:
        """
        Build token-optimized prompt for GPT-4o-mini.
        Focuses on problematic areas to reduce token usage.
        """
        
        paper_info = gpt_analysis_data.get('paper_info', {})
        structural_metrics = gpt_analysis_data.get('structural_metrics', {})
        problematic_areas = gpt_analysis_data.get('problematic_areas', [])
        focus_areas = gpt_analysis_data.get('analysis_focus', [])
        key_sections = gpt_analysis_data.get('key_sections', {})
        
        prompt_parts = []
        
        # Header
        prompt_parts.append("Analyze the coherence and cohesion of this scientific paper.\n")
        
        # Paper info
        prompt_parts.append(f"**Paper:** {paper_info.get('title', 'Untitled')[:200]}")
        prompt_parts.append(f"**Length:** {paper_info.get('word_count', 0)} words\n")
        
        # Structural analysis results
        prompt_parts.append("**STRUCTURAL ANALYSIS (Basic):**")
        prompt_parts.append(f"- Overall Score: {basic_score:.2f}/1.0")
        prompt_parts.append(f"- Connector Density: {structural_metrics.get('connector_density', 0):.1f} per 1000 words")
        prompt_parts.append(f"- Transition Ratio: {structural_metrics.get('transition_ratio', 0):.2f}")
        prompt_parts.append(f"- Terminology Issues: {structural_metrics.get('terminology_inconsistencies', 0)}")
        prompt_parts.append(f"- Structure Violations: {structural_metrics.get('order_violations', 0)}\n")
        
        # Focus areas
        if focus_areas:
            prompt_parts.append(f"**FOCUS ON:** {', '.join(focus_areas)}\n")
        
        # Key sections (limited to save tokens)
        if key_sections:
            prompt_parts.append("**KEY SECTIONS:**\n")
            # Only include first 2 sections or most relevant ones
            for i, (title, content) in enumerate(list(key_sections.items())[:2]):
                # Truncate content to ~500 chars
                truncated_content = content[:500] + "..." if len(content) > 500 else content
                prompt_parts.append(f"### {title}")
                prompt_parts.append(truncated_content + "\n")
        
        # Problematic areas (most important for focused analysis)
        if problematic_areas:
            prompt_parts.append("**PROBLEMATIC AREAS DETECTED:**\n")
            for i, area in enumerate(problematic_areas[:3], 1):  # Limit to 3
                if area['type'] == 'poor_transition':
                    prompt_parts.append(f"{i}. Poor transition at paragraph {area['paragraph_index']}:")
                    prompt_parts.append(f"   {area['text'][:300]}...\n")
                elif area['type'] == 'terminology_inconsistency':
                    prompt_parts.append(f"{i}. Terminology inconsistencies:")
                    for term1, term2 in area.get('examples', [])[:2]:
                        prompt_parts.append(f"   - '{term1}' vs '{term2}'")
                    prompt_parts.append("")
        
        # Analysis instructions
        prompt_parts.append("\n**YOUR TASK:**")
        prompt_parts.append("Provide a SEMANTIC and LOGICAL analysis focusing on:")
        prompt_parts.append("1. Are ideas truly connected, or just superficially linked?")
        prompt_parts.append("2. Does the argument flow logically from premise to conclusion?")
        prompt_parts.append("3. Are there logical gaps, circular reasoning, or non-sequiturs?")
        prompt_parts.append("4. Is the narrative consistent and coherent throughout?")
        prompt_parts.append("5. Specific problems in the identified problematic areas\n")
        
        # Required JSON format
        prompt_parts.append("**RESPOND IN THIS JSON FORMAT:**")
        prompt_parts.append("""{
  "overall_score": 0.85,
  "semantic_coherence": {
    "score": 0.8,
    "assessment": "Brief assessment"
  },
  "logical_flow": {
    "score": 0.9,
    "assessment": "Brief assessment"
  },
  "issues": [
    {
      "type": "logical_gap|poor_transition|inconsistency|other",
      "severity": "low|medium|high",
      "location": "Section or paragraph",
      "description": "Specific issue description",
      "example": "Quote or reference if applicable"
    }
  ],
  "suggestions": [
    {
      "priority": "high|medium|low",
      "action": "Specific actionable improvement",
      "rationale": "Why this improvement is needed"
    }
  ],
  "strengths": ["List of strong points"],
  "final_verdict": "EXCELLENT|GOOD|FAIR|POOR"
}""")
        
        return "\n".join(prompt_parts)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for the API call."""
        input_cost = (input_tokens / 1_000_000) * COST_PER_1M_INPUT
        output_cost = (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT
        return input_cost + output_cost
    
    def get_cost_report(self) -> Dict:
        """
        Generate detailed cost report.
        """
        avg_cost = self.total_cost / max(1, self.total_papers)
        avg_input_tokens = self.total_tokens_input / max(1, self.total_papers)
        avg_output_tokens = self.total_tokens_output / max(1, self.total_papers)
        
        return {
            'total_papers_analyzed': self.total_papers,
            'total_cost_usd': round(self.total_cost, 4),
            'average_cost_per_paper': round(avg_cost, 4),
            'total_tokens': {
                'input': self.total_tokens_input,
                'output': self.total_tokens_output,
                'total': self.total_tokens_input + self.total_tokens_output
            },
            'average_tokens_per_paper': {
                'input': round(avg_input_tokens),
                'output': round(avg_output_tokens)
            },
            'cost_projections': {
                '100_papers': round(avg_cost * 100, 2),
                '500_papers': round(avg_cost * 500, 2),
                '1000_papers': round(avg_cost * 1000, 2)
            },
            'analysis_history': self.analysis_history[-10:]  # Last 10 analyses
        }
    
    def format_gpt_feedback(self, gpt_result: Dict, basic_feedback: str) -> str:
        """
        Format GPT analysis into readable feedback to append to basic feedback.
        """
        if not gpt_result.get('success'):
            return basic_feedback + f" [GPT Analysis Failed: {gpt_result.get('error', 'Unknown error')}]"
        
        analysis = gpt_result.get('analysis', {})
        feedback_parts = [basic_feedback]
        
        # Add GPT score comparison
        gpt_score = analysis.get('overall_score', 0)
        feedback_parts.append(f"\n\n--- DEEP AI ANALYSIS (GPT-4o-mini) ---")
        feedback_parts.append(f"Enhanced Coherence Score: {gpt_score:.2f}/1.0")
        
        # Add semantic coherence
        if 'semantic_coherence' in analysis:
            sem_coh = analysis['semantic_coherence']
            feedback_parts.append(f"\nSemantic Coherence ({sem_coh.get('score', 0):.2f}): {sem_coh.get('assessment', 'N/A')}")
        
        # Add logical flow
        if 'logical_flow' in analysis:
            log_flow = analysis['logical_flow']
            feedback_parts.append(f"Logical Flow ({log_flow.get('score', 0):.2f}): {log_flow.get('assessment', 'N/A')}")
        
        # Add critical issues
        issues = analysis.get('issues', [])
        if issues:
            high_priority = [i for i in issues if i.get('severity') == 'high']
            if high_priority:
                feedback_parts.append(f"\n🔴 CRITICAL ISSUES ({len(high_priority)}):")
                for issue in high_priority[:3]:
                    feedback_parts.append(f"  • [{issue.get('type', 'unknown')}] {issue.get('description', 'N/A')}")
        
        # Add top suggestions
        suggestions = analysis.get('suggestions', [])
        if suggestions:
            high_priority_sugg = [s for s in suggestions if s.get('priority') == 'high']
            if high_priority_sugg:
                feedback_parts.append(f"\n💡 KEY RECOMMENDATIONS ({len(high_priority_sugg)}):")
                for sugg in high_priority_sugg[:3]:
                    feedback_parts.append(f"  • {sugg.get('action', 'N/A')}")
        
        # Add strengths
        strengths = analysis.get('strengths', [])
        if strengths:
            feedback_parts.append(f"\n✓ STRENGTHS: {', '.join(strengths[:3])}")
        
        # Add verdict
        verdict = analysis.get('final_verdict', 'N/A')
        feedback_parts.append(f"\nFinal Verdict: {verdict}")
        
        # Add cost info
        cost_info = gpt_result.get('cost_info', {})
        feedback_parts.append(f"\n[Analysis cost: ${cost_info.get('cost_usd', 0):.4f} | Tokens: {cost_info.get('total_tokens', 0)}]")
        
        return "\n".join(feedback_parts)


# ============== INTEGRATION FUNCTION ==============

def enhance_coherence_with_gpt(paper, basic_result: Dict, 
                               force_analysis: bool = False) -> Dict:
    """
    Main integration function: Enhance basic coherence analysis with GPT-4o-mini.
    
    Args:
        paper: Paper object
        basic_result: Result from check_cohesion.evaluate()
        force_analysis: Force GPT analysis even if score is good
    
    Returns:
        Enhanced result with GPT analysis
    """
    
    analyzer = GPTCoherenceAnalyzer()
    
    basic_score = basic_result.get('score', 0.0)
    basic_feedback = basic_result.get('feedback', '')
    gpt_analysis_data = basic_result.get('gpt_analysis_data', {})
    
    # Decide if GPT analysis is needed
    needs_gpt = force_analysis or analyzer.should_use_gpt_analysis(basic_score, basic_feedback)
    
    if not needs_gpt:
        # Return basic result with note
        basic_result['gpt_analysis'] = {
            'used': False,
            'reason': 'Basic analysis sufficient (score >= 0.7)',
            'cost_saved': 0.002  # Approximate savings
        }
        return basic_result
    
    # Perform GPT analysis
    gpt_result = analyzer.analyze_coherence(paper, gpt_analysis_data, basic_score)
    
    # Enhance feedback
    if gpt_result.get('success'):
        enhanced_feedback = analyzer.format_gpt_feedback(gpt_result, basic_feedback)
        basic_result['feedback'] = enhanced_feedback
        
        # Update score if GPT provides better assessment
        gpt_score = gpt_result.get('analysis', {}).get('overall_score')
        if gpt_score is not None:
            # Weighted average: 40% basic, 60% GPT
            basic_result['score'] = (basic_score * 0.4) + (gpt_score * 0.6)
            basic_result['score_breakdown'] = {
                'basic_score': basic_score,
                'gpt_score': gpt_score,
                'final_score': basic_result['score']
            }
    
    # Add GPT analysis to result
    basic_result['gpt_analysis'] = gpt_result
    basic_result['cost_report'] = analyzer.get_cost_report()
    
    return basic_result


# ============== USAGE EXAMPLE ==============

if __name__ == "__main__":
    """
    Example usage of the GPT integration module.
    """
    
    print("GPT-4o-mini Coherence Analyzer")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please add it to your .env file:")
        print('OPENAI_API_KEY="sk-your-key-here"')
        exit(1)
    
    print("✓ OpenAI API Key found")
    print(f"✓ Model: gpt-4o-mini")
    print(f"✓ Cost: ~${COST_PER_1M_INPUT:.2f} input + ${COST_PER_1M_OUTPUT:.2f} output per 1M tokens")
    print(f"✓ Estimated cost per paper: $0.002-0.004")
    print("\nReady to analyze papers!")