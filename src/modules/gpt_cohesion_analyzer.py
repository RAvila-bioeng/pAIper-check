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
from utils.semantic_preprocessor import extract_critical_snippets

# Load environment variables
load_dotenv()

# Cost tracking constants (per million tokens)
COST_PER_1M_INPUT = 0.15  # $0.15 per 1M input tokens
COST_PER_1M_OUTPUT = 0.60  # $0.60 per 1M output tokens


class GPTCoherenceAnalyzer:
    """
    Manages GPT-4o-mini analysis for deep coherence evaluation.
    Optimized for cost and quality.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
        Orchestrates a modular deep coherence analysis using GPT-4o-mini.
        """
        
        # --- Modular Analysis ---
        modules = {
            "terminology": self._analyze_terminology,
            "progression": self._analyze_progression,
            "abstract_expansion": self._analyze_abstract_expansion,
            "conclusion_response": self._analyze_conclusion_response,
            "lexical_richness": self._analyze_lexical_richness,
        }
        
        sub_results = {}
        total_cost = 0
        total_tokens = {'input': 0, 'output': 0}

        for name, module_func in modules.items():
            result = module_func(paper, gpt_analysis_data)
            sub_results[name] = result
            if result.get('success'):
                total_cost += result.get('cost_info', {}).get('cost_usd', 0)
                total_tokens['input'] += result.get('cost_info', {}).get('input_tokens', 0)
                total_tokens['output'] += result.get('cost_info', {}).get('output_tokens', 0)

        # --- Aggregation ---
        final_score, issues, suggestions, strengths = self._aggregate_results(sub_results, basic_score)

        final_analysis = {
            "overall_score": final_score,
            "sub_modules": {name: res.get('analysis', {}) for name, res in sub_results.items()},
            "issues": issues,
            "suggestions": suggestions,
            "strengths": strengths,
            "final_verdict": self._get_final_verdict(final_score)
        }
        
        # --- Cost Tracking ---
        self._update_cost_tracking(total_cost, total_tokens, gpt_analysis_data, final_score, basic_score)
        
        return {
            'success': True,
            'analysis': final_analysis,
            'cost_info': {
                'cost_usd': round(total_cost, 4),
                'input_tokens': total_tokens['input'],
                'output_tokens': total_tokens['output'],
                'total_tokens': total_tokens['input'] + total_tokens['output']
            },
            'model': 'gpt-4o-mini'
        }

    def _get_reflection_prompt(self) -> str:
        return """You are a meta-analyst AI. Your task is to critique and refine a previous analysis of a scientific paper.
Review the original request and the draft analysis. Identify weaknesses, inconsistencies, or overly generic feedback in the draft.
Then, generate a final, improved JSON response that is more accurate, specific, and reliable.
The final score should be a refined version of the draft score, taking into account your critical review.
Do not repeat the draft; provide a new, higher-quality JSON object."""

    def _call_gpt(self, prompt: str, system_prompt: str, max_tokens=300) -> Dict:
        """
        Helper function to make a structured, two-step (analysis + reflection) call to the OpenAI API.
        """
        total_cost = 0
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            # Step 1: Initial Analysis
            initial_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3, # Slightly higher temp for initial creative analysis
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            initial_analysis_text = initial_response.choices[0].message.content
            
            # Cost calculation for Step 1
            input_tokens_1 = initial_response.usage.prompt_tokens
            output_tokens_1 = initial_response.usage.completion_tokens
            total_cost += self._calculate_cost(input_tokens_1, output_tokens_1)
            total_input_tokens += input_tokens_1
            total_output_tokens += output_tokens_1

            # Step 2: Self-Reflection
            reflection_system_prompt = self._get_reflection_prompt()
            reflection_user_prompt = (
                "Please review and refine the following draft analysis.\n\n"
                "**Original Request:**\n"
                f"{prompt}\n\n"
                "**Draft Analysis (JSON):**\n"
                f"{initial_analysis_text}\n\n"
                "Now, provide your final, refined JSON analysis."
            )

            final_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": reflection_system_prompt},
                    {"role": "user", "content": reflection_user_prompt}
                ],
                temperature=0.1, # Lower temp for refinement
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )

            final_analysis_text = final_response.choices[0].message.content
            final_analysis_json = json.loads(final_analysis_text)

            # Cost calculation for Step 2
            input_tokens_2 = final_response.usage.prompt_tokens
            output_tokens_2 = final_response.usage.completion_tokens
            total_cost += self._calculate_cost(input_tokens_2, output_tokens_2)
            total_input_tokens += input_tokens_2
            total_output_tokens += output_tokens_2
            
            return {
                'success': True,
                'analysis': final_analysis_json,
                'cost_info': {
                    'cost_usd': round(total_cost, 4),
                    'input_tokens': total_input_tokens,
                    'output_tokens': total_output_tokens,
                }
            }
        except Exception as e:
            return {
                'success': False, 
                'error': str(e), 
                'analysis': {'score': 0.0, 'feedback': f'Analysis failed: {e}'},
                'cost_info': {'cost_usd': 0.0, 'input_tokens': 0, 'output_tokens': 0}
            }

    def _analyze_terminology(self, paper, gpt_analysis_data: Dict) -> Dict:
        """Analyzes the consistency of technical terms using semantic extraction."""
        system_prompt = "You are an expert scientific reviewer analyzing terminological consistency. Respond in JSON."
        
        # Extract critical snippets from the introduction, using the abstract as an anchor.
        introduction_content = paper.get_section_content('introduction')
        critical_intro = extract_critical_snippets(introduction_content, paper.abstract, num_snippets=5)

        prompt = (
            "Analyze the terminological consistency between the abstract and the key parts of the introduction.\n"
            f"**Abstract:**\n{paper.abstract}\n\n"
            f"**Key Introduction Snippets:**\n{critical_intro}\n\n"
            "YOUR TASK:\n"
            "1. Rate consistency from 0.0 to 1.0.\n"
            "2. Provide brief feedback on terminology usage.\n"
            "3. List up to 2 inconsistent term examples if any are found.\n"
            'RESPOND IN THIS JSON FORMAT:\n'
            '{"score": <float>, "feedback": "<string>", "inconsistencies": [{"term1": "<string>", "term2": "<string>"}]}'
        )
        return self._call_gpt(prompt, system_prompt)

    def _analyze_progression(self, paper, gpt_analysis_data: Dict) -> Dict:
        """Analyzes the logical flow between sections using semantic extraction."""
        system_prompt = "You are an expert scientific reviewer analyzing logical progression. Respond in JSON."
        
        # Extract critical snippets from each core section
        method_content = paper.get_section_content('methodology')
        results_content = paper.get_section_content('results')
        discussion_content = paper.get_section_content('discussion')

        critical_method = extract_critical_snippets(method_content, "Methodology", num_snippets=4)
        critical_results = extract_critical_snippets(results_content, "Results", num_snippets=4)
        critical_discussion = extract_critical_snippets(discussion_content, "Discussion", num_snippets=4)

        prompt = (
            "Analyze the logical progression between these key snippets from the paper's core sections.\n"
            f"**Key Methodology Snippets:**\n{critical_method}\n\n"
            f"**Key Results Snippets:**\n{critical_results}\n\n"
            f"**Key Discussion Snippets:**\n{critical_discussion}\n\n"
            "YOUR TASK:\n"
            "1. Rate the logical flow from 0.0 to 1.0 (do the results follow the methods? Does the discussion interpret the results?).\n"
            "2. Provide brief feedback on the narrative connection between these sections.\n"
            'RESPOND IN THIS JSON FORMAT:\n'
            '{"score": <float>, "feedback": "<string>"}'
        )
        return self._call_gpt(prompt, system_prompt)

    def _analyze_abstract_expansion(self, paper, gpt_analysis_data: Dict) -> Dict:
        """Checks if the introduction expands on the abstract using semantic extraction."""
        system_prompt = "You are an expert scientific reviewer analyzing the connection between abstract and introduction. Respond in JSON."
        
        introduction_content = paper.get_section_content('introduction')
        critical_intro = extract_critical_snippets(introduction_content, paper.abstract, num_snippets=5)

        prompt = (
            "Analyze if the key snippets from the introduction effectively expand on the promises made in the abstract.\n"
            f"**Abstract:**\n{paper.abstract}\n\n"
            f"**Key Introduction Snippets:**\n{critical_intro}\n\n"
            "YOUR TASK:\n"
            "1. Rate how well the introduction snippets expand on the abstract's core claims from 0.0 to 1.0.\n"
            "2. Provide brief feedback on this connection.\n"
            'RESPOND IN THIS JSON FORMAT:\n'
            '{"score": <float>, "feedback": "<string>"}'
        )
        return self._call_gpt(prompt, system_prompt)

    def _analyze_conclusion_response(self, paper, gpt_analysis_data: Dict) -> Dict:
        """Checks if the conclusion answers the initial questions using semantic extraction."""
        system_prompt = "You are an expert scientific reviewer analyzing the link between introduction goals and conclusion answers. Respond in JSON."
        
        # Extract critical snippets from both introduction and conclusion
        introduction_content = paper.get_section_content('introduction')
        conclusion_content = paper.get_section_content('conclusion')
        
        critical_intro = extract_critical_snippets(introduction_content, "Introduction objectives research questions", num_snippets=4)
        critical_conclusion = extract_critical_snippets(conclusion_content, "Conclusion summary findings", num_snippets=4)

        prompt = (
            "Analyze if the key snippets from the conclusion effectively address the objectives stated in the key snippets from the introduction.\n"
            f"**Key Introduction Snippets (Objectives):**\n{critical_intro}\n\n"
            f"**Key Conclusion Snippets (Answers):**\n{critical_conclusion}\n\n"
            "YOUR TASK:\n"
            "1. Rate how well the conclusion snippets answer the introduction's stated goals from 0.0 to 1.0.\n"
            "2. Provide brief feedback on this narrative closure.\n"
            'RESPOND IN THIS JSON FORMAT:\n'
            '{"score": <float>, "feedback": "<string>"}'
        )
        return self._call_gpt(prompt, system_prompt)

    def _analyze_lexical_richness(self, paper, gpt_analysis_data: Dict) -> Dict:
        """Analyzes word repetition and vocabulary variety using a semantic sample."""
        system_prompt = "You are an expert linguistic reviewer analyzing lexical richness. Respond in JSON."
        
        # Use semantic extraction to get a diverse sample of the text, rather than just the start.
        # We use the paper's title as a broad anchor to get a general sense of the content.
        text_sample = extract_critical_snippets(paper.full_text, paper.title, num_snippets=10)
        
        # Fallback if sample is too short
        if len(text_sample) < 500:
            text_sample = paper.full_text[:3000]

        prompt = (
            "Analyze the lexical richness of the following representative text sample. Look for excessive repetition of non-technical words and phrases.\n"
            f"**Text Sample:**\n{text_sample}...\n\n"
            "YOUR TASK:\n"
            "1. Rate the lexical richness from 0.0 (very repetitive) to 1.0 (very rich).\n"
            "2. Provide brief feedback on vocabulary use.\n"
            "3. List up to 2 examples of repetitive non-technical phrases if any are found.\n"
            'RESPOND IN THIS JSON FORMAT:\n'
            '{"score": <float>, "feedback": "<string>", "repetitive_phrases": ["<phrase1>", "<phrase2>"]}'
        )
        return self._call_gpt(prompt, system_prompt, max_tokens=400)

    def _aggregate_results(self, sub_results: Dict, basic_score: float) -> tuple:
        """Aggregates results from all sub-modules."""
        issues, suggestions, strengths = [], [], []
        
        weights = {"terminology": 0.25, "progression": 0.30, "abstract_expansion": 0.20, "conclusion_response": 0.20, "lexical_richness": 0.05}
        
        total_score, total_weight = 0, 0
        
        for name, result in sub_results.items():
            if result.get('success'):
                analysis = result.get('analysis', {})
                score = analysis.get('score', 0.0)
                feedback = analysis.get('feedback', 'No feedback.')
                
                total_score += score * weights.get(name, 0)
                total_weight += weights.get(name, 0)
                
                if score < 0.6:
                    issues.append(f"Weak {name.replace('_', ' ')}: {feedback}")
                elif score >= 0.8:
                    strengths.append(f"Strong {name.replace('_', ' ')}: {feedback}")

        final_score = (total_score / total_weight) if total_weight > 0 else basic_score
        
        return final_score, issues, suggestions, strengths

    def _get_final_verdict(self, score: float) -> str:
        if score >= 0.85: return "EXCELLENT"
        if score >= 0.7: return "GOOD"
        if score >= 0.5: return "FAIR"
        return "POOR"

    def _update_cost_tracking(self, cost, tokens, gpt_analysis_data, gpt_score, basic_score):
        self.total_cost += cost
        self.total_papers += 1
        self.total_tokens_input += tokens['input']
        self.total_tokens_output += tokens['output']
        self.analysis_history.append({
            'paper_title': gpt_analysis_data.get('paper_info', {}).get('title', 'Unknown'),
            'basic_score': basic_score,
            'gpt_score': gpt_score,
            'cost': cost,
            'tokens': tokens,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
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
        
        # Add sub-module scores
        if 'sub_modules' in analysis:
            feedback_parts.append("\nCohesion Sub-modules:")
            for name, sub in analysis['sub_modules'].items():
                feedback_parts.append(f"  - {name.replace('_', ' ').title()}: {sub.get('score', 0):.2f} - {sub.get('feedback', 'N/A')}")

        # Add critical issues
        issues = analysis.get('issues', [])
        if issues:
            feedback_parts.append(f"\n🔴 CRITICAL ISSUES ({len(issues)}):")
            for issue in issues[:3]:
                feedback_parts.append(f"  • {issue}")
        
        # Add top suggestions
        suggestions = analysis.get('suggestions', [])
        if suggestions:
            feedback_parts.append(f"\n💡 KEY RECOMMENDATIONS ({len(suggestions)}):")
            for sugg in suggestions[:3]:
                feedback_parts.append(f"  • {sugg}")
        
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
