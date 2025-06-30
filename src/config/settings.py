"""Configuration settings for Legend QA Extractor."""

import os
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List


@dataclass
class Config:
    """Configuration class for QA Extractor."""
    
    # File paths
    pdf_filename: str = "uploaded.pdf"
    output_filename: str = "output_final_qa.jsonl"
    output_dir: str = "output"
    
    # Model configuration
    model_name: str = "qwen2.5:7b-instruct"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.1
    
    # Text processing
    max_block_size: int = 1500
    min_block_size: int = 100
    extract_ratio: float = 1.0
    
    # QA filtering
    enable_qa_filter: bool = False
    known_prefixes: List[str] = None
    
    # Logging
    log_level: str = "INFO"
    enable_success_log: bool = True
    enable_error_log: bool = True
    
    def __post_init__(self):
        if self.known_prefixes is None:
            self.known_prefixes = [
                "网友", "记者", "问", "提问者", "主持人", 
                "文章引用", "Q", "观众", "评论", "主持", "用户"
            ]


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file and environment variables.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded settings
    """
    config = Config()
    
    # Load from YAML file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                for key, value in yaml_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
    
    # Override with environment variables
    env_mappings = {
        'PDF_FILENAME': 'pdf_filename',
        'OUTPUT_FILENAME': 'output_filename',
        'OUTPUT_DIR': 'output_dir',
        'OLLAMA_MODEL': 'model_name',
        'OLLAMA_HOST': 'ollama_host',
        'MODEL_TEMPERATURE': 'temperature',
        'MAX_BLOCK_SIZE': 'max_block_size',
        'MIN_BLOCK_SIZE': 'min_block_size',
        'EXTRACT_RATIO': 'extract_ratio',
        'ENABLE_QA_FILTER': 'enable_qa_filter',
        'LOG_LEVEL': 'log_level',
    }
    
    for env_var, config_attr in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Type conversion
            if config_attr in ['temperature', 'extract_ratio']:
                env_value = float(env_value)
            elif config_attr in ['max_block_size', 'min_block_size']:
                env_value = int(env_value)
            elif config_attr in ['enable_qa_filter']:
                env_value = env_value.lower() in ('true', '1', 'yes', 'on')
            
            setattr(config, config_attr, env_value)
    
    return config


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Config object to save
        config_path: Path to save the configuration file
    """
    config_dict = asdict(config)
    
    # Create directory if it doesn't exist
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def get_default_config_path() -> str:
    """Get the default configuration file path."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml') 