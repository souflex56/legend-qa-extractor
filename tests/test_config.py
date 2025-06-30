"""Tests for configuration module."""

import os
import tempfile
import pytest
from unittest.mock import patch

from src.config import Config, load_config, save_config


class TestConfig:
    """Test cases for Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.pdf_filename == "uploaded.pdf"
        assert config.output_filename == "output_final_qa.jsonl"
        assert config.output_dir == "output"
        assert config.model_name == "qwen2.5:7b-instruct"
        assert config.ollama_host == "http://localhost:11434"
        assert config.temperature == 0.1
        assert config.max_block_size == 1500
        assert config.min_block_size == 100
        assert config.extract_ratio == 1.0
        assert config.enable_qa_filter is False
        assert config.log_level == "INFO"
        assert config.enable_success_log is True
        assert config.enable_error_log is True
    
    def test_known_prefixes_default(self):
        """Test default known prefixes are set correctly."""
        config = Config()
        
        expected_prefixes = [
            "网友", "记者", "问", "提问者", "主持人", 
            "文章引用", "Q", "观众", "评论", "主持", "用户"
        ]
        
        assert config.known_prefixes == expected_prefixes
    
    def test_custom_known_prefixes(self):
        """Test custom known prefixes."""
        custom_prefixes = ["测试1", "测试2"]
        config = Config(known_prefixes=custom_prefixes)
        
        assert config.known_prefixes == custom_prefixes


class TestConfigIO:
    """Test cases for configuration I/O operations."""
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        config = Config()
        config.pdf_filename = "test.pdf"
        config.model_name = "test-model"
        config.temperature = 0.5
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            # Save config
            save_config(config, config_path)
            assert os.path.exists(config_path)
            
            # Load config
            loaded_config = load_config(config_path)
            
            assert loaded_config.pdf_filename == "test.pdf"
            assert loaded_config.model_name == "test-model"
            assert loaded_config.temperature == 0.5
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        config = load_config("nonexistent_file.yaml")
        
        # Should return default config
        assert isinstance(config, Config)
        assert config.pdf_filename == "uploaded.pdf"
    
    @patch.dict(os.environ, {
        'PDF_FILENAME': 'env_test.pdf',
        'MODEL_TEMPERATURE': '0.8',
        'MAX_BLOCK_SIZE': '2000',
        'ENABLE_QA_FILTER': 'true'
    })
    def test_environment_variable_override(self):
        """Test environment variable override."""
        config = load_config()
        
        assert config.pdf_filename == "env_test.pdf"
        assert config.temperature == 0.8
        assert config.max_block_size == 2000
        assert config.enable_qa_filter is True
    
    def test_invalid_yaml_file(self):
        """Test handling of invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            # Should not raise exception, should return default config
            config = load_config(config_path)
            assert isinstance(config, Config)
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path) 