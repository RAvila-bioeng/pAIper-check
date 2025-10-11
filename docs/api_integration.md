# API Integration

## OpenAI API
Used to perform linguistic, cohesion, and quality evaluations.

### Environment Variables
- `OPENAI_API_KEY`: required
- `MODEL_NAME`: optional, defaults to `gpt-4`

### Example
```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze this paragraph..."}]
)
