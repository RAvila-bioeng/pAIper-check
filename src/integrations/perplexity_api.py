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

# --- ANÁLISIS DE REFERENCIAS ---
def analyze_references(references):
    """
    Analiza las referencias de un artículo utilizando Perplexity Sonar Pro.
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
    - Un resumen general de la calidad de las referencias.
    - Puntos fuertes específicos.
    - Áreas de mejora con sugerencias concretas.
    """
    
    # Formatear las referencias para el prompt
    references_text = "\n".join([f"- {ref.text}" for ref in references])
    user_prompt = f"Analiza la siguiente lista de referencias:\n{references_text}"
    
    return call_perplexity_api(system_prompt, user_prompt)

# --- ANÁLISIS DE ESTRUCTURA ---
def analyze_structure(paper_structure):
    """
    Analiza la estructura del artículo utilizando Perplexity Sonar Pro.
    """
    system_prompt = """
    Eres un editor académico experimentado. Tu tarea es evaluar la estructura y organización
    de un artículo científico basándote en las secciones detectadas.
    
    Criterios de evaluación:
    1.  **Completitud:** ¿Están presentes las secciones canónicas (Introducción, Metodología, Resultados, Conclusión, Referencias)?
    2.  **Orden lógico:** ¿Siguen las secciones una progresión lógica y coherente?
    3.  **Claridad:** ¿Son los títulos de las secciones claros y descriptivos?
    
    Formato de salida:
    - Evaluación general de la estructura.
    - Puntos fuertes de la organización.
    - Sugerencias para mejorar la estructura.
    """
    
    structure_text = "\n".join([f"- {section.title}" for section in paper_structure])
    user_prompt = f"Evalúa la estructura del artículo basándote en estas secciones:\n{structure_text}"
    
    return call_perplexity_api(system_prompt, user_prompt)

# --- ANÁLISIS LINGÜÍSTICO ---
def analyze_linguistics(paper_text):
    """
    Realiza un análisis lingüístico avanzado con Perplexity Sonar Pro.
    """
    system_prompt = """
    Eres un experto en comunicación científica y lingüística. Tu tarea es evaluar la calidad
    del lenguaje de un extracto de texto académico.
    
    Criterios de evaluación:
    1.  **Claridad y Precisión:** ¿Es el lenguaje claro, preciso y sin ambigüedades?
    2.  **Tono Académico:** ¿Se mantiene un tono formal y objetivo?
    3.  **Complejidad:** ¿Es la sintaxis adecuada para un texto científico (ni demasiado simple ni innecesariamente compleja)?
    
    Formato de salida:
    - Feedback general sobre la calidad lingüística.
    - Aspectos positivos destacados.
    - Recomendaciones para mejorar la redacción.
    """
    
    # Usar un extracto para no exceder los límites de la API
    excerpt = paper_text[:4000]
    user_prompt = f"Analiza la calidad lingüística del siguiente texto:\n\n{excerpt}"
    
    return call_perplexity_api(system_prompt, user_prompt)

# --- ANÁLISIS DE REPRODUCIBILIDAD ---
def analyze_reproducibility(paper_text):
    """
    Evalúa la reproducibilidad del estudio con Perplexity Sonar Pro.
    """
    system_prompt = """
    Eres un revisor científico especializado en reproducibilidad. Tu tarea es evaluar
    si un estudio parece reproducible a partir de su descripción.
    
    Criterios de evaluación:
    1.  **Detalle Metodológico:** ¿Se describen los métodos con suficiente detalle para que otro investigador pueda replicarlos?
    2.  **Disponibilidad de Datos y Código:** ¿Se menciona si los datos, el código o los materiales están disponibles?
    3.  **Transparencia:** ¿Se discuten las limitaciones y los parámetros de manera transparente?
    
    Formato de salida:
    - Veredicto general sobre la reproducibilidad.
    - Indicadores positivos de reproducibilidad encontrados.
    - Sugerencias para mejorar la transparencia y la reproducibilidad.
    """
    
    # Usar un extracto relevante (Metodología y Resultados)
    methodology_match = re.search(r"Methodology\s*(.*?)\s*Results", paper_text, re.DOTALL | re.IGNORECASE)
    if methodology_match:
        excerpt = methodology_match.group(1)
    else:
        excerpt = paper_text[:4000]

    user_prompt = f"Evalúa la reproducibilidad basándote en este texto:\n\n{excerpt}"
    
    return call_perplexity_api(system_prompt, user_prompt)


# --- FUNCIÓN GENÉRICA DE LLAMADA A LA API ---
def call_perplexity_api(system_prompt, user_prompt):
    """
    Realiza una llamada a la API de Perplexity con los prompts de sistema y de usuario.
    Devuelve un diccionario estructurado con el análisis o un error.
    """
    if not API_KEY:
        return {
            "success": False,
            "error": "PERPLEXITY_API_KEY no encontrada en las variables de entorno."
        }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "sonar-pro",
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
        
        return {
            "success": True,
            "analysis": analysis_content
        }

    except requests.exceptions.RequestException as e:
        error_message = f"Error en la llamada a la API de Perplexity: {e}"
        # Intentar obtener más detalles del error si es posible
        try:
            error_details = e.response.json()
            error_message += f" | Detalles: {error_details}"
        except:
            pass
            
        return {
            "success": False,
            "error": error_message
        }
    except (KeyError, IndexError) as e:
        return {
            "success": False,
            "error": f"Respuesta inesperada de la API de Perplexity: {e}"
        }
