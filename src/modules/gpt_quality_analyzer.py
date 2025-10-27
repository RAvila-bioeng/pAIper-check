"""
GPT Integration for Scientific Quality Evaluation.
This module provides a deep, AI-based analysis of a paper's scientific quality.
"""

import os
import json
from typing import Dict

# Optional GPT integration
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False


if GPT_AVAILABLE:
    class GPTQualityAnalyzer:
        """GPT-4o-mini integration for deep scientific quality analysis."""
        
        def __init__(self):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.total_cost = 0.0
            self.total_papers = 0
        
        def should_use_gpt_analysis(self, basic_score: float, feedback: str) -> bool:
            """Decide if GPT analysis is needed based on basic analysis results."""
            if basic_score < 0.7:
                return True
            
            critical_keywords = [
                'needs improvement', 'could be enhanced', 'missing',
                'insufficient', 'limited', 'weak'
            ]
            
            feedback_lower = feedback.lower()
            if any(keyword in feedback_lower for keyword in critical_keywords):
                return True
            
            return False
        
        def analyze_quality(self, paper, gpt_analysis_data: Dict, basic_score: float) -> Dict:
            """Perform deep scientific quality analysis using GPT-4o-mini."""
            
            modules = {
                "novelty_assessment": self._analyze_novelty,
                "rigor_evaluation": self._analyze_rigor,
                "significance_analysis": self._analyze_significance,
                "theoretical_contribution": self._analyze_theory,
                "practical_impact": self._analyze_practical
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
            
            # Aggregate results
            final_score, issues, suggestions, strengths = self._aggregate_results(sub_results, basic_score)
            
            final_analysis = {
                "overall_score": final_score,
                "sub_modules": {name: res.get('analysis', {}) for name, res in sub_results.items()},
                "issues": issues,
                "suggestions": suggestions,
                "strengths": strengths,
                "final_verdict": self._get_final_verdict(final_score)
            }
            
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
        
        def _analyze_novelty(self, paper, gpt_analysis_data: Dict) -> Dict:
            """Analyze novelty and originality using GPT."""
            system_prompt = "You are an expert scientific reviewer analyzing novelty and originality. Respond in JSON."
            
            prompt = (
                f"Analyze the novelty and originality of this research paper.\n"
                f"**Title:** {paper.title}\n"
                f"**Abstract:** {paper.abstract}\n\n"
                "YOUR TASK:\n"
                "1. Rate novelty from 0.0 to 1.0.\n"
                "2. Identify specific novel contributions.\n"
                "3. Assess how the work advances the field.\n"
                "4. Note any gaps in novelty claims.\n"
                'RESPOND IN JSON FORMAT:\n'
                '{"score": <float>, "novel_contributions": ["<contribution1>", "<contribution2>"], "field_advancement": "<string>", "gaps": ["<gap1>", "<gap2>"]}'
            )
            return self._call_gpt(prompt, system_prompt)
        
        def _analyze_rigor(self, paper, gpt_analysis_data: Dict) -> Dict:
            """Analyze methodological rigor using GPT."""
            system_prompt = "You are an expert scientific reviewer analyzing methodological rigor. Respond in JSON."
            
            methodology_content = paper.get_section_content('methodology') or "Methodology section not found"
            
            prompt = (
                f"Analyze the methodological rigor of this research.\n"
                f"**Title:** {paper.title}\n"
                f"**Methodology:** {methodology_content[:1000]}...\n\n"
                "YOUR TASK:\n"
                "1. Rate methodological rigor from 0.0 to 1.0.\n"
                "2. Assess experimental design quality.\n"
                "3. Evaluate statistical analysis appropriateness.\n"
                "4. Identify methodological limitations.\n"
                'RESPOND IN JSON FORMAT:\n'
                '{"score": <float>, "design_quality": "<string>", "statistical_appropriateness": "<string>", "limitations": ["<limitation1>", "<limitation2>"]}'
            )
            return self._call_gpt(prompt, system_prompt)
        
        def _analyze_significance(self, paper, gpt_analysis_data: Dict) -> Dict:
            """Analyze results significance using GPT."""
            system_prompt = "You are an expert scientific reviewer analyzing results significance. Respond in JSON."
            
            results_content = paper.get_section_content('results') or "Results section not found"
            
            prompt = (
                f"Analyze the significance of the research results.\n"
                f"**Title:** {paper.title}\n"
                f"**Results:** {results_content[:1000]}...\n\n"
                "YOUR TASK:\n"
                "1. Rate results significance from 0.0 to 1.0.\n"
                "2. Assess quantitative evidence strength.\n"
                "3. Evaluate comparison with baselines.\n"
                "4. Identify significance limitations.\n"
                'RESPOND IN JSON FORMAT:\n'
                '{"score": <float>, "evidence_strength": "<string>", "baseline_comparisons": "<string>", "limitations": ["<limitation1>", "<limitation2>"]}'
            )
            return self._call_gpt(prompt, system_prompt)
        
        def _analyze_theory(self, paper, gpt_analysis_data: Dict) -> Dict:
            """Analyze theoretical contribution using GPT."""
            system_prompt = "You are an expert scientific reviewer analyzing theoretical contributions. Respond in JSON."
            
            prompt = (
                f"Analyze the theoretical contribution of this research.\n"
                f"**Title:** {paper.title}\n"
                f"**Abstract:** {paper.abstract}\n\n"
                "YOUR TASK:\n"
                "1. Rate theoretical contribution from 0.0 to 1.0.\n"
                "2. Identify theoretical frameworks used.\n"
                "3. Assess theoretical implications.\n"
                "4. Note theoretical limitations.\n"
                'RESPOND IN JSON FORMAT:\n'
                '{"score": <float>, "frameworks": ["<framework1>", "<framework2>"], "implications": "<string>", "limitations": ["<limitation1>", "<limitation2>"]}'
            )
            return self._call_gpt(prompt, system_prompt)
        
        def _analyze_practical(self, paper, gpt_analysis_data: Dict) -> Dict:
            """Analyze practical implications using GPT."""
            system_prompt = "You are an expert scientific reviewer analyzing practical implications. Respond in JSON."
            
            prompt = (
                f"Analyze the practical implications of this research.\n"
                f"**Title:** {paper.title}\n"
                f"**Abstract:** {paper.abstract}\n\n"
                "YOUR TASK:\n"
                "1. Rate practical implications from 0.0 to 1.0.\n"
                "2. Identify potential applications.\n"
                "3. Assess broader impact.\n"
                "4. Note practical limitations.\n"
                'RESPOND IN JSON FORMAT:\n'
                '{"score": <float>, "applications": ["<app1>", "<app2>"], "broader_impact": "<string>", "limitations": ["<limitation1>", "<limitation2>"]}'
            )
            return self._call_gpt(prompt, system_prompt)
        
        def _call_gpt(self, prompt: str, system_prompt: str, max_tokens=400) -> Dict:
            """Make a structured call to the OpenAI API."""
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                )
                
                analysis_text = response.choices[0].message.content
                analysis_json = json.loads(analysis_text)
                
                cost = self._calculate_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                
                return {
                    'success': True,
                    'analysis': analysis_json,
                    'cost_info': {
                        'cost_usd': round(cost, 4),
                        'input_tokens': response.usage.prompt_tokens,
                        'output_tokens': response.usage.completion_tokens,
                    }
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'analysis': {'score': 0.0, 'feedback': f'Analysis failed: {e}'},
                    'cost_info': {'cost_usd': 0.0, 'input_tokens': 0, 'output_tokens': 0}
                }
        
        def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
            """Calculate cost in USD for the API call."""
            COST_PER_1M_INPUT = 0.15
            COST_PER_1M_OUTPUT = 0.60
            input_cost = (input_tokens / 1_000_000) * COST_PER_1M_INPUT
            output_cost = (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT
            return input_cost + output_cost
        
        def _aggregate_results(self, sub_results: Dict, basic_score: float) -> tuple:
            """Aggregate results from all sub-modules."""
            issues, suggestions, strengths = [], [], []
            
            weights = {"novelty_assessment": 0.25, "rigor_evaluation": 0.20, "significance_analysis": 0.20, 
                      "theoretical_contribution": 0.15, "practical_impact": 0.20}
            
            total_score, total_weight = 0, 0
            
            for name, result in sub_results.items():
                if result.get('success'):
                    analysis = result.get('analysis', {})
                    score = analysis.get('score', 0.0)
                    
                    total_score += score * weights.get(name, 0)
                    total_weight += weights.get(name, 0)
                    
                    if score < 0.6:
                        issues.append(f"Weak {name.replace('_', ' ')}: {analysis.get('feedback', 'No feedback')}")
                    elif score >= 0.8:
                        strengths.append(f"Strong {name.replace('_', ' ')}: {analysis.get('feedback', 'No feedback')}")
            
            final_score = (total_score / total_weight) if total_weight > 0 else basic_score
            
            return final_score, issues, suggestions, strengths
        
        def _get_final_verdict(self, score: float) -> str:
            if score >= 0.85: return "EXCELLENT"
            if score >= 0.7: return "GOOD"
            if score >= 0.5: return "FAIR"
            return "POOR"

    def enhance_quality_with_gpt(paper, basic_result: Dict, force_analysis: bool = False) -> Dict:
        """Main integration function: Enhance basic quality analysis with GPT-4o-mini."""
        
        analyzer = GPTQualityAnalyzer()
        
        basic_score = basic_result.get('score', 0.0)
        basic_feedback = basic_result.get('feedback', '')
        gpt_analysis_data = basic_result.get('gpt_analysis_data', {})
        
        # Decide if GPT analysis is needed
        needs_gpt = force_analysis

        
        if not needs_gpt:
            basic_result['gpt_analysis'] = {
                'used': False,
                'reason': 'Basic analysis sufficient (score >= 0.7)',
                'cost_saved': 0.003
            }
            return basic_result
        
        # Perform GPT analysis
        gpt_result = analyzer.analyze_quality(paper, gpt_analysis_data, basic_score)
        
        # Enhance feedback
        if gpt_result.get('success'):
            enhanced_feedback = format_gpt_feedback(gpt_result, basic_feedback)
            basic_result['feedback'] = enhanced_feedback
            
            # Update score if GPT provides better assessment
            gpt_score = gpt_result.get('analysis', {}).get('overall_score')
            if gpt_score is not None:
                # Weighted average: 40% basic, 60% GPT
                basic_result['score'] = (basic_score * 0.4) + (gpt_score * 0.6)
                basic_result['score_breakdown']['gpt_enhanced'] = True
        
        # Add GPT analysis to result
        basic_result['gpt_analysis'] = gpt_result
        
        return basic_result

    def format_gpt_feedback(gpt_result: Dict, basic_feedback: str) -> str:
        """Format GPT analysis into readable feedback."""
        if not gpt_result.get('success'):
            return basic_feedback + f" [GPT Analysis Failed: {gpt_result.get('error', 'Unknown error')}]"
        
        analysis = gpt_result.get('analysis', {})
        feedback_parts = [basic_feedback]
        
        # Add GPT score comparison
        gpt_score = analysis.get('overall_score', 0)
        feedback_parts.append(f"\n\n--- DEEP AI ANALYSIS (GPT-4o-mini) ---")
        feedback_parts.append(f"Enhanced Quality Score: {gpt_score:.2f}/1.0")
        
        # Add sub-module scores
        if 'sub_modules' in analysis:
            feedback_parts.append("\nQuality Sub-modules:")
            for name, sub in analysis['sub_modules'].items():
                feedback_parts.append(f"  - {name.replace('_', ' ').title()}: {sub.get('score', 0):.2f}")
        
        # Add critical issues
        issues = analysis.get('issues', [])
        if issues:
            feedback_parts.append(f"\n🔴 CRITICAL ISSUES ({len(issues)}):")
            for issue in issues[:3]:
                feedback_parts.append(f"  • {issue}")
        
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
else:
    # Define dummy functions if GPT is not available
    def enhance_quality_with_gpt(paper, basic_result: Dict, force_analysis: bool = False) -> Dict:
        basic_result['gpt_analysis'] = {'used': False, 'reason': 'GPT dependencies not installed.'}
        return basic_result
