"""LLM client module for interacting with Ollama models."""

from typing import Optional, Dict, Any
import logging
from ollama import Client

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with Ollama LLM models."""
    
    def __init__(self, host: str = "http://localhost:11434", model_name: str = "qwen2.5:7b-instruct"):
        self.host = host
        self.model_name = model_name
        self.client = Client(host=host)
        self.logger = logger
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to list models to test connection
            models = self.client.list()
            self.logger.info(f"Successfully connected to Ollama at {self.host}")
            model_list = models.get('models', []) if isinstance(models, dict) else []
            model_names = [m.get('name', 'unknown') for m in model_list if isinstance(m, dict)]
            self.logger.info(f"Available models: {model_names}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama at {self.host}: {e}")
            return False
    
    def call_ollama(self, prompt: str, temperature: float = 0.1, **kwargs) -> Optional[str]:
        """Call Ollama model with given prompt.
        
        Args:
            prompt: Input prompt for the model
            temperature: Model temperature for response generation
            **kwargs: Additional options for the model
            
        Returns:
            Model response text or None if failed
        """
        try:
            options = {
                "temperature": temperature,
                **kwargs
            }
            
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=options
            )
            
            if response and "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                self.logger.warning("Unexpected response format from Ollama")
                return None
                
        except Exception as e:
            self.logger.error(f"Ollama API call failed: {e}")
            return None
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current model.
        
        Returns:
            Model information dictionary or None if failed
        """
        try:
            models = self.client.list()
            model_list = models.get('models', []) if isinstance(models, dict) else []
            for model in model_list:
                if isinstance(model, dict) and model.get('name') == self.model_name:
                    return model
            return None
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return None
    
    def check_model_availability(self) -> bool:
        """Check if the specified model is available.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            models = self.client.list()
            model_list = models.get('models', []) if isinstance(models, dict) else []
            model_names = [m.get('name', 'unknown') for m in model_list if isinstance(m, dict)]
            available = self.model_name in model_names
            
            if not available:
                self.logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
            
            return available
        except Exception as e:
            self.logger.error(f"Failed to check model availability: {e}")
            return False
    
    def pull_model(self) -> bool:
        """Pull the model if not available locally.
        
        Returns:
            True if model pull successful, False otherwise
        """
        try:
            self.logger.info(f"Pulling model {self.model_name}...")
            self.client.pull(self.model_name)
            self.logger.info(f"Successfully pulled model {self.model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to pull model {self.model_name}: {e}")
            return False
    
    def set_model(self, model_name: str) -> bool:
        """Set the model to use.
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            True if model is set and available, False otherwise
        """
        old_model = self.model_name
        self.model_name = model_name
        
        if self.check_model_availability():
            self.logger.info(f"Model changed from {old_model} to {model_name}")
            return True
        else:
            self.model_name = old_model  # Revert on failure
            return False 