import yaml
import os
from dotenv import load_dotenv

def load_config(config_path: str = "config/default.yaml") -> dict:
    """
    Loads configuration values from a YAML file and environment variables.
    """
    load_dotenv()  # Carga variables del .env

    # Carga YAML
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # Fusiona con entorno (entorno tiene prioridad)
    config.update({
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", config.get("OPENAI_API_KEY")),
        "MONGO_URI": os.getenv("MONGO_URI", config.get("MONGO_URI")),
        "DB_NAME": os.getenv("DB_NAME", config.get("DB_NAME", "pAIperCheck")),
    })

    return config
