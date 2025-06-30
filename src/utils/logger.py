"""Logging utilities for Legend QA Extractor."""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logger(name: str, 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 enable_console: bool = True,
                 log_format: Optional[str] = None) -> logging.Logger:
    """Set up a logger with specified configuration.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_console: Whether to enable console logging
        log_format: Custom log format string
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_specialized_logger(name: str, 
                           log_file: str,
                           log_format: str,
                           log_level: str = "INFO") -> logging.Logger:
    """Set up a specialized logger for specific purposes (e.g., error logs, success logs).
    
    Args:
        name: Logger name
        log_file: Path to log file
        log_format: Log format string
        log_level: Logging level
        
    Returns:
        Configured specialized logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Ensure directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_extraction_loggers(output_dir: str) -> tuple[logging.Logger, logging.Logger]:
    """Set up specialized loggers for extraction process.
    
    Args:
        output_dir: Directory for log files
        
    Returns:
        Tuple of (error_logger, success_logger)
    """
    error_log_path = os.path.join(output_dir, "extraction_errors.log")
    success_log_path = os.path.join(output_dir, "extraction_success.log")
    
    # Error logger
    error_format = '%(asctime)s - %(levelname)s - BLOCK_START\n%(message)s\nBLOCK_END'
    error_logger = setup_specialized_logger(
        'error_logger', 
        error_log_path, 
        error_format
    )
    
    # Success logger
    success_format = '%(asctime)s - %(levelname)s - QA_PAIR_START\n%(message)s\nQA_PAIR_END'
    success_logger = setup_specialized_logger(
        'success_logger', 
        success_log_path, 
        success_format
    )
    
    return error_logger, success_logger 