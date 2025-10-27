import requests
import os

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_ENDPOINT = "https://api.perplexity.ai/chat/completions"

def analyze_references(references):
    if not PERPLEXITY_API_KEY:
        raise ValueError("❌ Missing PERPLEXITY_API_KEY environment variable")

    refs_text = "\n".join([r.text for r in references if r.text])

    prompt = (
        "Analiza las siguientes referencias académicas y clasifícalas según su calidad, "
        "actualidad, relevancia y accesibilidad. Resume si son buenas, mejorables o desactualizadas.\n\n"
        f"{refs_text}"
    )

    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": prompt}]
    }

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(PERPLEXITY_ENDPOINT, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]