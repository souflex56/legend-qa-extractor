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
    max_block_size: int = 1000
    min_block_size: int = 200
    extract_ratio: float = 1.0
    
    # Semantic grouping configuration
    semantic_grouping: dict = None
    
    # Long answer processing configuration
    long_answer_processing: dict = None
    
    # LLM anchor for Q&A pairs
    enable_llm_anchor: bool = False
    anchor_keywords_count: int = 3
    
    # Token management configuration
    max_prompt_tokens: int = 6000
    enable_token_monitoring: bool = True  # 启用自动token监控和报告
    
    # QA filtering
    enable_qa_filter: bool = True
    skip_low_confidence: bool = False
    known_prefixes: List[str] = None
    
    # Logging
    log_level: str = "INFO"
    enable_success_log: bool = True
    enable_error_log: bool = True
    
    # **🚀 PERFORMANCE OPTIMIZATION: Batch processing configuration**
    batch_size: int = 5
    max_workers: int = 3
    
    def __post_init__(self):
        if self.known_prefixes is None:
            self.known_prefixes = [
                "网友", "记者", "问", "提问者", "主持人", 
                "文章引用", "Q", "观众", "评论", "主持", "用户"
            ]
        
        # Set default semantic grouping configuration
        if self.semantic_grouping is None:
            self.semantic_grouping = {
                'max_question_length': 50,
                'default_similarity_threshold': 0.65,
                'std_factor': 0.5
            }
        
        # Set default long answer processing configuration
        if self.long_answer_processing is None:
            self.long_answer_processing = {
                'chain_summary_threshold': 3000,
                'summary_length': 50,
                'nli_model_path': 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
                'entailment_threshold': 0.7
            }
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)


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
        'ENABLE_LLM_ANCHOR': 'enable_llm_anchor',
        'ANCHOR_KEYWORDS_COUNT': 'anchor_keywords_count',
        'ENABLE_QA_FILTER': 'enable_qa_filter',
        'LOG_LEVEL': 'log_level',
        'BATCH_SIZE': 'batch_size',
        'MAX_WORKERS': 'max_workers',
    }
    
    for env_var, config_attr in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Type conversion
            if config_attr in ['temperature', 'extract_ratio', 'qa_allowance_ratio']:
                env_value = float(env_value)
            elif config_attr in ['max_block_size', 'min_block_size', 'anchor_keywords_count', 'batch_size', 'max_workers']:
                env_value = int(env_value)
            elif config_attr in ['enable_qa_filter', 'enable_sliding_context', 'enable_llm_anchor']:
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