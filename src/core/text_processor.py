"""Text processing module for handling text segmentation and cleaning."""

import re
from typing import List
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text processing, segmentation, and cleaning operations."""
    
    def __init__(self, known_prefixes: List[str] = None):
        self.logger = logger
        self.known_prefixes = known_prefixes or [
            "网友", "记者", "问", "提问者", "主持人", 
            "文章引用", "Q", "观众", "评论", "主持", "用户"
        ]
    
    def preprocess_qa_text(self, text: str) -> str:
        """Preprocess text to standardize Q&A formats.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text with standardized formats
        """
        if not text:
            return ""
        
        # Standardize Q&A identifiers
        text = re.sub(r'网友[：:]', '网友：', text)
        text = re.sub(r'问[：:]', '问：', text)
        text = re.sub(r'段永平[：:]', '段永平：', text)
        text = re.sub(r'段[：:]', '段：', text)
        text = re.sub(r'大道[：:]', '大道：', text)
        
        # Clean extra blank lines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text
    
    def clean_question_text(self, question: str) -> str:
        """Remove question prefixes from question text.
        
        Args:
            question: Question text with potential prefixes
            
        Returns:
            Cleaned question text without prefixes
        """
        if question is None:
            return ""
        
        question = str(question)
        
        # Known prefixes pattern
        pattern = r'^({})[：:]\s*'.format('|'.join(re.escape(p) for p in self.known_prefixes))
        cleaned = re.sub(pattern, '', question).strip()
        if cleaned != question:
            return cleaned

        # Fallback pattern for unknown prefixes
        fallback_pattern = r'^[\u4e00-\u9fa5A-Za-z0-9（）【】「」《》''""、,，.。·\s]{1,20}[：:]\s*'
        return re.sub(fallback_pattern, '', question).strip()
    
    def create_hybrid_blocks(self, text: str, max_size: int, min_size: int) -> List[str]:
        """Create text blocks using hybrid natural paragraph merging strategy.
        
        Args:
            text: Input text to segment
            max_size: Maximum block size in characters
            min_size: Minimum block size in characters
            
        Returns:
            List of text blocks
        """
        if not text:
            return []
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        all_blocks = []
        i = 0

        while i < len(paragraphs):
            block_paras = []
            current_len = 0

            while i < len(paragraphs):
                para = paragraphs[i]
                # Check if adding this paragraph would exceed max_size
                if current_len > 0 and current_len + len(para) > max_size:
                    break
                
                block_paras.append(para)
                current_len += len(para)
                i += 1

                # If we've reached or exceeded max_size, break
                if current_len >= max_size:
                    break

            if block_paras:  # Only create block if we have paragraphs
                block_text = "\n\n".join(block_paras)
                if len(block_text) >= min_size:
                    all_blocks.append(block_text)

        self.logger.info(f"Created {len(all_blocks)} text blocks")
        return all_blocks
    
    def block_has_qa(self, text: str) -> bool:
        """Check if text block contains Q&A patterns.
        
        Args:
            text: Text block to analyze
            
        Returns:
            True if block contains Q&A patterns, False otherwise
        """
        if not text:
            return False
        
        # Direct question patterns
        direct_question_patterns = [
            r"网友[：:]",
            r"问[：:]",
            r"问题[：:]", 
            r"提问[：:]",
            r"主持人[：:]",
            r"观众[：:]",
            r"Q[：:]"
        ]
        
        # Indirect question patterns
        indirect_question_patterns = [
            r"文章引用[：:]",
            r"引用[：:]",
            r"有人说",
            r"有人认为",
            r"有观点认为",
            r"据说",
            r"听说"
        ]
        
        # Answer patterns
        answer_patterns = [
            r"段永平[：:]",
            r"段[：:]",
            r"大道[：:]"
        ]
        
        has_direct_question = any(re.search(p, text) for p in direct_question_patterns)
        has_indirect_question = any(re.search(p, text) for p in indirect_question_patterns)
        has_answer = any(re.search(p, text) for p in answer_patterns)
        
        return bool(has_answer and (has_direct_question or has_indirect_question))
    
    def filter_blocks_by_ratio(self, blocks: List[str], ratio: float) -> List[str]:
        """Filter blocks by sampling ratio.
        
        Args:
            blocks: List of text blocks
            ratio: Sampling ratio (0.0 to 1.0)
            
        Returns:
            Filtered list of blocks
        """
        if ratio >= 1.0:
            return blocks
        
        if ratio <= 0.0:
            return []
        
        sample_size = max(int(len(blocks) * ratio), 1)
        filtered_blocks = blocks[:sample_size]
        
        self.logger.info(f"Filtered {len(filtered_blocks)} blocks from {len(blocks)} (ratio: {ratio:.1%})")
        return filtered_blocks
    
    def validate_text_quality(self, text: str) -> bool:
        """Validate text quality for processing.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text quality is acceptable, False otherwise
        """
        if not text or not text.strip():
            return False
        
        # Check minimum length
        if len(text.strip()) < 10:
            return False
        
        # Check if text has readable content (not just symbols or numbers)
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        if chinese_chars < len(text) * 0.1:  # At least 10% Chinese characters
            return False
        
        return True 