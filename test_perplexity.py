"""Test simple de Perplexity sin imports complejos"""
import os
import sys
import requests
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PERPLEXITY_API_KEY")
API_URL = "https://api.perplexity.ai/chat/completions"

print("="*70)
print("🧪 TEST DE PERPLEXITY API")
print("="*70)

# 1. Verificar API Key
if not API_KEY:
    print("❌ PERPLEXITY_API_KEY no encontrada en .env")
    print("\nAñade a tu archivo .env:")
    print("PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxx")
    sys.exit(1)

print(f"✓ API Key encontrada: {API_KEY[:20]}...")

# 2. Test de conexión
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "sonar",  # ✅ CAMBIADO: modelo correcto
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello from Perplexity' in one sentence."}
    ]
}

print("\n🔍 Probando conexión con Perplexity API...")
print(f"   URL: {API_URL}")
print(f"   Model: sonar")

try:
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    
    # Verificar status code
    if response.status_code != 200:
        print(f"\n❌ Error HTTP {response.status_code}")
        print(f"   Response: {response.text}")
        sys.exit(1)
    
    data = response.json()
    
    # Extraer información
    content = data['choices'][0]['message']['content']
    usage = data['usage']
    
    print("\n✅ CONEXIÓN EXITOSA!")
    print("="*70)
    print(f"\n📝 Respuesta de Perplexity:")
    print(f"   {content}")
    print(f"\n📊 Uso de tokens:")
    print(f"   Input:  {usage['prompt_tokens']}")
    print(f"   Output: {usage['completion_tokens']}")
    print(f"   Total:  {usage['total_tokens']}")
    
    # Calcular costo (nuevo pricing)
    input_cost = (usage['prompt_tokens'] / 1_000_000) * 1.00
    output_cost = (usage['completion_tokens'] / 1_000_000) * 1.00
    total_cost = input_cost + output_cost
    
    print(f"\n💰 Costo estimado: ${total_cost:.6f}")
    
    print("\n" + "="*70)
    print("✅ Perplexity API está funcionando correctamente")
    print("="*70)
    
except requests.exceptions.Timeout:
    print("\n❌ Timeout - La API no respondió a tiempo")
    sys.exit(1)
    
except requests.exceptions.RequestException as e:
    print(f"\n❌ Error de conexión: {e}")
    sys.exit(1)
    
except KeyError as e:
    print(f"\n❌ Respuesta inesperada de la API")
    print(f"   Falta el campo: {e}")
    print(f"   Respuesta completa: {json.dumps(data, indent=2)}")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Error inesperado: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)