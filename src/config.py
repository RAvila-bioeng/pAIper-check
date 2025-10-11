import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import logging

load_dotenv()


class Config:
    """Configuration management for pAIper check."""
    
    def __init__(self, config_file: str = "config/default.yaml"):
        self.config_file = config_file
        self._config_data = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        config_path = Path(self.config_file)
        
        if not config_path.exists():
            logging.warning(f"Config file {self.config_file} not found, using defaults")
            self._config_data = self._get_default_config()
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                self._config_data = yaml.safe_load(file) or {}
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            self._config_data = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'app': {
                'debug': False,
                'log_level': 'INFO'
            },
            'llm': {
                'model_name': 'gpt-4',
                'temperature': 0.2,
                'max_tokens': 2000
            },
            'database': {
                'uri': '',
                'name': 'pAIperCheck'
            },
            'evaluation': {
                'weights': {
                    'structure': 0.15,
                    'linguistics': 0.15,
                    'cohesion': 0.15,
                    'reproducibility': 0.20,
                    'references': 0.15,
                    'quality': 0.20
                },
                'min_score_threshold': 0.6
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'app.debug')."""
        keys = key_path.split('.')
        value = self._config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment or config."""
        return os.getenv("OPENAI_API_KEY") or self.get('llm.api_key')
    
    def get_mongo_uri(self) -> str:
        """Get MongoDB URI from environment or config."""
        return os.getenv("MONGO_URI") or self.get('database.uri', '')
    
    def get_db_name(self) -> str:
        """Get database name from environment or config."""
        return os.getenv("DB_NAME") or self.get('database.name', 'pAIperCheck')
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return os.getenv("DEBUG", "False").lower() == "true" or self.get('app.debug', False)
    
    def get_log_level(self) -> str:
        """Get logging level."""
        return os.getenv("LOG_LEVEL") or self.get('app.log_level', 'INFO')
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return {
            'model_name': self.get('llm.model_name', 'gpt-4'),
            'temperature': self.get('llm.temperature', 0.2),
            'max_tokens': self.get('llm.max_tokens', 2000),
            'api_key': self.get_openai_api_key()
        }
    
    def get_evaluation_weights(self) -> Dict[str, float]:
        """Get evaluation weights for each pillar."""
        return self.get('evaluation.weights', {
            'structure': 0.15,
            'linguistics': 0.15,
            'cohesion': 0.15,
            'reproducibility': 0.20,
            'references': 0.15,
            'quality': 0.20
        })
    
    def validate(self) -> bool:
        """Validate configuration."""
        errors = []
        
        # Check required API key
        if not self.get_openai_api_key():
            errors.append("OpenAI API key is required")
        
        # Check weights sum to 1.0
        weights = self.get_evaluation_weights()
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Evaluation weights must sum to 1.0, got {total_weight}")
        
        if errors:
            for error in errors:
                logging.error(f"Config validation error: {error}")
            return False
        
        return True


def setup_logging(config: Optional[Config] = None):
    """Setup logging configuration."""
    if config is None:
        config = Config()
    
    log_level = getattr(logging, config.get_log_level().upper(), logging.INFO)
    
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=log_level,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('paiper_check.log')
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


# Global config instance
config = Config()
