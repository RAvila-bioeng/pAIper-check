# pAIper Check - Guía de Uso

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Configurar variables de entorno (opcional):
```bash
# Crear archivo .env
OPENAI_API_KEY=tu_api_key_aqui
MONGO_URI=tu_mongo_uri_aqui
DEBUG=false
LOG_LEVEL=INFO
```

## Uso Básico

### Evaluar un paper
```bash
python src/main.py --input data/sample_paper_1.txt
```

### Evaluar con salida verbose
```bash
python src/main.py --input data/sample_paper_1.txt --verbose
```

### Guardar resultados en archivo JSON
```bash
python src/main.py --input data/sample_paper_1.txt --output results.json
```

## Módulos de Evaluación

El sistema evalúa papers en 6 pilares principales:

### 1. Structure & Completeness (15%)
- Verifica presencia de secciones esenciales
- Evalúa estructura del documento
- Analiza calidad del título

### 2. Linguistic Quality (15%)
- Detección de errores ortográficos
- Consistencia terminológica
- Estilo académico apropiado
- Patrones gramaticales

### 3. Coherence & Cohesion (15%)
- Fluidez argumentativa
- Conectividad entre secciones
- Consistencia narrativa
- Flujo lógico

### 4. Reproducibility (20%)
- Claridad metodológica
- Disponibilidad de datos
- Disponibilidad de código
- Especificación de parámetros

### 5. References & Citations (15%)
- Formato de citas
- Calidad de referencias
- Densidad de citas
- Recencia de referencias

### 6. Scientific Quality (20%)
- Novedad y originalidad
- Rigor metodológico
- Significado de resultados
- Contribución teórica

## Configuración

El archivo `config/default.yaml` permite personalizar:

- Pesos de evaluación por pilar
- Configuración del LLM
- Niveles de logging
- Configuración de base de datos

## Testing

Ejecutar tests:
```bash
pytest tests/
```

## Estructura del Proyecto

```
src/
├── models/           # Modelos de datos
├── modules/          # Módulos de evaluación
├── utils/            # Utilidades (parser PDF, etc.)
├── config.py         # Sistema de configuración
└── main.py          # Punto de entrada principal

data/
├── sample_paper_1.txt    # Paper de muestra
├── sample_paper_2.txt    # Paper de muestra
└── ground_trouth_scores.csv  # Scores de referencia

config/
└── default.yaml      # Configuración por defecto
```

## Formato de Salida

El sistema genera un reporte con:
- Score general (0.0 - 1.0)
- Scores por pilar
- Feedback detallado
- Información del paper

Ejemplo:
```
============================================================
📊 pAIper Check Evaluation Results
============================================================
📄 Paper: Machine Learning Applications in Healthcare
🎯 Overall Score: 0.78/1.0

📋 Pillar Scores:
  • Structure & Completeness: 0.80 (weight: 0.15)
  • Linguistic Quality: 0.70 (weight: 0.15)
  • Coherence & Cohesion: 0.75 (weight: 0.15)
  • Reproducibility: 0.60 (weight: 0.20)
  • References & Citations: 0.85 (weight: 0.15)
  • Scientific Quality: 0.80 (weight: 0.20)

💬 Feedback Summary:
  • Structure & Completeness: Good structure with all essential sections present.
  • Linguistic Quality: Some informal language detected. Maintain formal academic tone.
  • Coherence & Cohesion: Good coherence with logical flow and consistent narrative.
  • Reproducibility: Data availability information is missing. Specify how data can be accessed.
  • References & Citations: Good reference quality with appropriate citation density.
  • Scientific Quality: Strong scientific quality with clear contributions and rigorous methodology.

✅ Evaluation complete!
```

## Próximos Pasos

- Integración con MongoDB para almacenamiento persistente
- Interfaz web para evaluación interactiva
- API REST para integración con otros sistemas
- Mejoras en el parser de PDF para mayor precisión
- Integración con LLMs para evaluación más avanzada
