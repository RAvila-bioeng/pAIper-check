import os
import re
import requests
import json
from dotenv import load_dotenv

# --- CONFIGURACIÓN DE LA API ---
API_URL = "https://api.perplexity.ai/chat/completions"
# Cargar la clave de la API desde el archivo .env
load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Configuración de precios (Perplexity Sonar - actualizar según pricing oficial)
# https://docs.perplexity.ai/docs/pricing
PRICING = {
    "sonar-small-32k-online": {
        "input_per_1m": 0.20,   # $0.20 por 1M tokens de entrada
        "output_per_1m": 0.20   # $0.20 por 1M tokens de salida
    },
    "sonar-pro": {
        "input_per_1m": 3.00,   # $3.00 por 1M tokens de entrada
        "output_per_1m": 15.00  # $15.00 por 1M tokens de salida
    }
}

# --- ANÁLISIS DE REFERENCIAS ---
def analyze_references(references):
    """
    Analiza las referencias de un artículo utilizando Perplexity Sonar.
    
    Returns:
        dict: Estructura con success, analysis, model, cost_info, y error (si aplica)
    """
    system_prompt = """
    Eres un experto en metodología de la investigación y análisis bibliométrico.
    Tu tarea es evaluar la calidad y relevancia de la sección de referencias de un artículo científico.
    Analiza la siguiente lista de referencias y proporciona un feedback conciso y constructivo.
    
    Criterios de evaluación:
    1.  **Relevancia:** ¿Son las fuentes pertinentes para el tema del artículo?
    2.  **Actualidad:** ¿Hay un buen equilibrio entre referencias clásicas y recientes (últimos 5 años)?
    3.  **Calidad:** ¿Provienen las fuentes de revistas, conferencias o editoriales de prestigio?
    4.  **Diversidad:** ¿Se citan diferentes tipos de fuentes (artículos, libros, conferencias)?
    
    Formato de salida:
    - Un resumen general de la calidad de las referencias (2-3 líneas).
    - Puntos fuertes específicos (2-3 puntos).
    - Áreas de mejora con sugerencias concretas (2-3 puntos).
    """
    
    # Formatear las referencias para el prompt (limitar a primeras 50 para no exceder límites)
    references_sample = references[:50]
    references_text = "\n".join([f"- {ref.text}" for ref in references_sample])
    
    if len(references) > 50:
        references_text += f"\n\n... y {len(references) - 50} referencias más."
    
    user_prompt = f"Analiza la siguiente lista de {len(references)} referencias:\n\n{references_text}"
    
    return call_perplexity_api(system_prompt, user_prompt, model="sonar-small-32k-online")


# --- ANÁLISIS DE ESTRUCTURA ---
def analyze_structure(paper_structure):
    """
    Analiza la estructura del artículo utilizando Perplexity Sonar.
    
    Returns:
        dict: Estructura con success, analysis, model, cost_info, y error (si aplica)
    """
    system_prompt = """
    Eres un editor académico experimentado. Tu tarea es evaluar la estructura y organización
    de un artículo científico basándote en las secciones detectadas.
    
    Criterios de evaluación:
    1.  **Completitud:** ¿Están presentes las secciones canónicas (Introducción, Metodología, Resultados, Conclusión, Referencias)?
    2.  **Orden lógico:** ¿Siguen las secciones una progresión lógica y coherente?
    3.  **Claridad:** ¿Son los títulos de las secciones claros y descriptivos?
    
    Formato de salida:
    - Evaluación general de la estructura (2-3 líneas).
    - Puntos fuertes de la organización (2-3 puntos).
    - Sugerencias para mejorar la estructura (2-3 puntos).
    """
    
    structure_text = "\n".join([f"- {section.title}" for section in paper_structure])
    user_prompt = f"Evalúa la estructura del artículo basándote en estas secciones:\n\n{structure_text}"
    
    return call_perplexity_api(system_prompt, user_prompt, model="sonar-small-32k-online")


# --- ANÁLISIS LINGÜÍSTICO ---
def analyze_linguistics(paper_text):
    """
    Realiza un análisis lingüístico avanzado con Perplexity Sonar.
    
    Returns:
        dict: Estructura con success, analysis, model, cost_info, y error (si aplica)
    """
    system_prompt = """
    Eres un experto en comunicación científica y lingüística. Tu tarea es evaluar la calidad
    del lenguaje de un extracto de texto académico.
    
    Criterios de evaluación:
    1.  **Claridad y Precisión:** ¿Es el lenguaje claro, preciso y sin ambigüedades?
    2.  **Tono Académico:** ¿Se mantiene un tono formal y objetivo?
    3.  **Complejidad:** ¿Es la sintaxis adecuada para un texto científico (ni demasiado simple ni innecesariamente compleja)?
    
    Formato de salida:
    - Feedback general sobre la calidad lingüística (2-3 líneas).
    - Aspectos positivos destacados (2-3 puntos).
    - Recomendaciones para mejorar la redacción (2-3 puntos).
    """
    
    # Usar un extracto para no exceder los límites de la API
    excerpt = paper_text[:4000]
    user_prompt = f"Analiza la calidad lingüística del siguiente texto:\n\n{excerpt}"
    
    return call_perplexity_api(system_prompt, user_prompt, model="sonar-small-32k-online")


# --- ANÁLISIS DE REPRODUCIBILIDAD ---
def analyze_reproducibility(paper_text):
    """
    Evalúa la reproducibilidad del estudio con Perplexity Sonar.
    
    Returns:
        dict: Estructura con success, analysis, model, cost_info, y error (si aplica)
    """
    system_prompt = """
    Eres un revisor científico especializado en reproducibilidad. Tu tarea es evaluar
    si un estudio parece reproducible a partir de su descripción.
    
    Criterios de evaluación:
    1.  **Detalle Metodológico:** ¿Se describen los métodos con suficiente detalle para que otro investigador pueda replicarlos?
    2.  **Disponibilidad de Datos y Código:** ¿Se menciona si los datos, el código o los materiales están disponibles?
    3.  **Transparencia:** ¿Se discuten las limitaciones y los parámetros de manera transparente?
    
    Formato de salida:
    - Veredicto general sobre la reproducibilidad (2-3 líneas).
    - Indicadores positivos de reproducibilidad encontrados (2-3 puntos).
    - Sugerencias para mejorar la transparencia y la reproducibilidad (2-3 puntos).
    """
    
    # Usar un extracto relevante (Metodología y Resultados)
    methodology_match = re.search(r"Methodology\s*(.*?)\s*Results", paper_text, re.DOTALL | re.IGNORECASE)
    if methodology_match:
        excerpt = methodology_match.group(1)[:4000]
    else:
        excerpt = paper_text[:4000]

    user_prompt = f"Evalúa la reproducibilidad basándote en este texto:\n\n{excerpt}"
    
    return call_perplexity_api(system_prompt, user_prompt, model="sonar-small-32k-online")


# --- FUNCIÓN GENÉRICA DE LLAMADA A LA API ---
def call_perplexity_api(system_prompt, user_prompt, model="sonar-small-32k-online"):
    """
    Realiza una llamada a la API de Perplexity con los prompts de sistema y de usuario.
    
    Args:
        system_prompt: Instrucciones del sistema
        user_prompt: Prompt del usuario
        model: Modelo a usar (default: sonar-small-32k-online)
    
    Returns:
        dict: Estructura estandarizada con:
            - success: bool
            - analysis: str (contenido del análisis)
            - model: str (modelo usado)
            - cost_info: dict (con cost_usd, input_tokens, output_tokens, total_tokens)
            - error: str (solo si success=False)
    """
    if not API_KEY:
        return {
            "success": False,
            "error": "PERPLEXITY_API_KEY no encontrada en las variables de entorno. Añádela en tu archivo .env",
            "model": model,
            "cost_info": {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()  # Lanza un error para respuestas 4xx/5xx
        
        data = response.json()
        
        # Extraer el contenido del análisis
        analysis_content = data['choices'][0]['message']['content']
        
        # Extraer información de uso (tokens)
        usage = data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
        
        # Calcular costo
        cost_usd = _calculate_cost(model, input_tokens, output_tokens)
        
        return {
            "success": True,
            "analysis": analysis_content,
            "model": model,
            "cost_info": {
                "cost_usd": round(cost_usd, 4),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        }

    except requests.exceptions.RequestException as e:
        error_message = f"Error en la llamada a la API de Perplexity: {str(e)}"
        
        # Intentar obtener más detalles del error si es posible
        try:
            if hasattr(e, 'response') and e.response is not None:
                error_details = e.response.json()
                error_message += f" | Detalles: {error_details}"
        except:
            pass
            
        return {
            "success": False,
            "error": error_message,
            "model": model,
            "cost_info": {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }
        
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {
            "success": False,
            "error": f"Respuesta inesperada de la API de Perplexity: {str(e)}",
            "model": model,
            "cost_info": {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calcula el costo en USD de una llamada a la API.
    
    Args:
        model: Modelo usado
        input_tokens: Tokens de entrada
        output_tokens: Tokens de salida
    
    Returns:
        float: Costo en USD
    """
    pricing = PRICING.get(model, PRICING["sonar-small-32k-online"])
    
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_1m"]
    
    return input_cost + output_cost