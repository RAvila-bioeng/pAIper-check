"""
Demo: Showing the difference between structural analysis (FREE) and GPT analysis (REQUIRES API)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from modules.check_cohesion import evaluate, generate_gpt_prompt, get_gpt_analysis_summary
from utils.pdf_parser import parse_paper_from_pdf


def demo_structural_analysis_only():
    """Demo showing what works WITHOUT API key."""
    
    print("="*70)
    print("DEMO: ANÁLISIS ESTRUCTURAL (GRATIS) - SIN API KEY")
    print("="*70)
    
    # Parse paper
    paper_path = Path(__file__).parent.parent / 'data' / 'sample_paper_1.txt'
    paper = parse_paper_from_pdf(str(paper_path))
    
    # Run structural analysis (this works without API)
    result = evaluate(paper)
    
    print(f"RESULTADO DEL ANALISIS ESTRUCTURAL:")
    print(f"   Score: {result['score']:.2f}/1.0")
    print(f"   Feedback: {result['feedback']}")
    print()
    
    # Show GPT readiness data (this also works without API)
    gpt_data = result.get('gpt_analysis_data', {})
    
    if gpt_data:
        print("DATOS PREPARADOS PARA GPT-4o-mini:")
        summary = get_gpt_analysis_summary(gpt_data)
        print(f"   {summary}")
        print()
        
        print("PROMPT GENERADO PARA GPT:")
        prompt = generate_gpt_prompt(gpt_data)
        print("   (Primeras 200 caracteres del prompt)")
        print(f"   {prompt[:200]}...")
        print()
        
        print("PARA ANALISIS GPT REAL:")
        print("   1. Necesitas configurar OPENAI_API_KEY en .env")
        print("   2. Llamar a analyze_with_gpt(gpt_data)")
        print("   3. GPT analizara los problemas especificos identificados")
        print()
        
        # Show what would happen with GPT analysis
        print("LO QUE HARIA GPT-4o-mini:")
        focus_areas = gpt_data.get('analysis_focus', [])
        problematic_count = len(gpt_data.get('problematic_areas', []))
        
        print(f"   - Enfocaria en: {', '.join(focus_areas)}")
        print(f"   - Analizaria {problematic_count} areas problematicas especificas")
        print(f"   - Proporcionaria sugerencias detalladas y actionable")
        print(f"   - Daria un score mas preciso basado en analisis profundo")
        print()
        
        print("VENTAJA DE LA ESTRATEGIA:")
        print("   - Analisis estructural: GRATIS y RAPIDO")
        print("   - Identificacion de problemas: AUTOMATICA")
        print("   - GPT analisis: SOLO cuando es necesario y enfocado")
        print("   - Costo total: MINIMO (solo tokens para problemas especificos)")


def demo_with_api_key():
    """Demo showing what would happen WITH API key."""
    
    print("="*70)
    print("DEMO: ANÁLISIS CON GPT-4o-mini (REQUIERE API KEY)")
    print("="*70)
    
    print("CON API KEY CONFIGURADA:")
    print("   from modules.gpt_cohesion_analyzer import analyze_with_gpt")
    print("   gpt_result = analyze_with_gpt(gpt_data)")
    print()
    
    print("RESULTADO ESPERADO DEL ANALISIS GPT:")
    print("   {")
    print("     'score': 0.75,  # Score mas preciso")
    print("     'issues': [")
    print("       'Poor transition between paragraphs 2 and 3',")
    print("       'Inconsistent terminology usage in methodology section',")
    print("       'Lack of logical connectors in discussion'")
    print("     ],")
    print("     'suggestions': [")
    print("       'Add transitional phrases like \"Furthermore\" or \"In contrast\"',")
    print("       'Standardize terminology: use either \"machine learning\" or \"ML\" consistently',")
    print("       'Include more causal connectors to improve argument flow'")
    print("     ],")
    print("     'severity': 'medium'")
    print("   }")
    print()
    
    print("COSTO ESTIMADO:")
    print("   - Prompt optimizado: ~800 tokens")
    print("   - Respuesta GPT: ~400 tokens")
    print("   - Total: ~1200 tokens")
    print("   - Costo GPT-4o-mini: ~$0.0003 por analisis")
    print("   - Analisis de 100 papers: ~$0.03")


if __name__ == "__main__":
    demo_structural_analysis_only()
    print("\n" + "="*70)
    demo_with_api_key()
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("- El sistema funciona PERFECTAMENTE sin API key")
    print("- Proporciona analisis estructural detallado (GRATIS)")
    print("- Prepara datos optimizados para GPT (LISTO)")
    print("- GPT analisis es OPCIONAL y BARATO cuando se necesita")
    print("="*70)
