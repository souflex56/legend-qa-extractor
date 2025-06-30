"""File handling utilities for Legend QA Extractor."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> str:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        The directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_jsonl(data: List[Dict[str, Any]], file_path: str, append: bool = False) -> None:
    """Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to output file
        append: Whether to append to existing file
    """
    # Ensure directory exists
    ensure_dir(os.path.dirname(file_path))
    
    mode = 'a' if append else 'w'
    
    try:
        with open(file_path, mode, encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        action = "Appended" if append else "Saved"
        logger.info(f"{action} {len(data)} items to {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")
        raise


def save_single_jsonl_item(item: Dict[str, Any], file_path: str) -> None:
    """Save a single item to JSONL file (append mode).
    
    Args:
        item: Dictionary to save
        file_path: Path to output file
    """
    # Ensure directory exists
    ensure_dir(os.path.dirname(file_path))
    
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"Failed to save item to {file_path}: {e}")
        raise


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries loaded from file
    """
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return []
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                        continue
        
        logger.info(f"Loaded {len(data)} items from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        return []


def iter_jsonl(file_path: str) -> Iterator[Dict[str, Any]]:
    """Iterate over JSONL file items without loading all into memory.
    
    Args:
        file_path: Path to JSONL file
        
    Yields:
        Dictionary items from the file
    """
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                        continue
    except Exception as e:
        logger.error(f"Failed to read from {file_path}: {e}")


def get_file_size(file_path: str) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes, 0 if file doesn't exist
    """
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0


def count_jsonl_lines(file_path: str) -> int:
    """Count the number of lines in a JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        Number of lines in the file
    """
    if not os.path.exists(file_path):
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        logger.error(f"Failed to count lines in {file_path}: {e}")
        return 0


def backup_file(file_path: str, backup_suffix: str = ".bak") -> Optional[str]:
    """Create a backup of the specified file.
    
    Args:
        file_path: Path to file to backup
        backup_suffix: Suffix for backup file
        
    Returns:
        Path to backup file if successful, None otherwise
    """
    if not os.path.exists(file_path):
        logger.warning(f"Cannot backup non-existent file: {file_path}")
        return None
    
    backup_path = file_path + backup_suffix
    
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup of {file_path}: {e}")
        return None


def cleanup_temp_files(directory: str, pattern: str = "*.tmp") -> int:
    """Clean up temporary files in the specified directory.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match (glob pattern)
        
    Returns:
        Number of files cleaned up
    """
    import glob
    
    if not os.path.exists(directory):
        return 0
    
    pattern_path = os.path.join(directory, pattern)
    temp_files = glob.glob(pattern_path)
    
    cleaned_count = 0
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            cleaned_count += 1
        except Exception as e:
            logger.warning(f"Failed to remove temp file {temp_file}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} temporary files from {directory}")
    
    return cleaned_count 