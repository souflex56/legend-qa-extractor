"""Configuration management for Legend QA Extractor."""

from .settings import Config, load_config, save_config, get_default_config_path

__all__ = ["Config", "load_config", "save_config", "get_default_config_path"] 