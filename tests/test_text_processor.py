"""Tests for text processing module."""

import pytest
from src.core.text_processor import TextProcessor


class TestTextProcessor:
    """Test cases for TextProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TextProcessor()
    
    def test_preprocess_qa_text(self):
        """Test Q&A text preprocessing."""
        input_text = "网友：这是测试\n段永平：这是回答\n\n\n额外空行"
        expected = "网友：这是测试\n段永平：这是回答\n\n额外空行"
        
        result = self.processor.preprocess_qa_text(input_text)
        assert result == expected
    
    def test_clean_question_text(self):
        """Test question text cleaning."""
        test_cases = [
            ("网友：什么是投资？", "什么是投资？"),
            ("问：如何学习？", "如何学习？"),
            ("Q：这是问题吗？", "这是问题吗？"),
            ("已经清洁的问题", "已经清洁的问题"),
            ("", ""),
            (None, ""),
        ]
        
        for input_text, expected in test_cases:
            result = self.processor.clean_question_text(input_text)
            assert result == expected, f"Failed for input: {input_text}"
    
    def test_create_hybrid_blocks_basic(self):
        """Test basic block creation."""
        text = "段落1\n\n段落2\n\n段落3"
        blocks = self.processor.create_hybrid_blocks(text, max_size=20, min_size=5)
        
        assert len(blocks) > 0
        for block in blocks:
            assert len(block) >= 5  # min_size
    
    def test_create_hybrid_blocks_empty_text(self):
        """Test block creation with empty text."""
        blocks = self.processor.create_hybrid_blocks("", max_size=100, min_size=10)
        assert blocks == []
    
    def test_create_hybrid_blocks_size_constraints(self):
        """Test block creation respects size constraints."""
        text = "短\n\n" + "很长的段落" * 100
        blocks = self.processor.create_hybrid_blocks(text, max_size=50, min_size=10)
        
        for block in blocks:
            assert len(block) >= 10  # min_size constraint
    
    def test_block_has_qa_positive_cases(self):
        """Test Q&A detection for positive cases."""
        positive_cases = [
            "网友：问题\n段永平：回答",
            "问：什么是价值投资？\n段：这是一个好问题",
            "文章引用：某某说了什么\n大道：我觉得这个观点有道理",
            "有人认为投资很难\n段永平：确实不容易",
        ]
        
        for text in positive_cases:
            assert self.processor.block_has_qa(text), f"Failed for: {text}"
    
    def test_block_has_qa_negative_cases(self):
        """Test Q&A detection for negative cases."""
        negative_cases = [
            "只是普通文本",
            "网友：问题但没有回答",
            "段永平：回答但没有问题",
            "方丈：这不是段永平的回答",
            "",
        ]
        
        for text in negative_cases:
            assert not self.processor.block_has_qa(text), f"Failed for: {text}"
    
    def test_filter_blocks_by_ratio(self):
        """Test block filtering by ratio."""
        blocks = [f"Block {i}" for i in range(10)]
        
        # Test various ratios
        assert len(self.processor.filter_blocks_by_ratio(blocks, 1.0)) == 10
        assert len(self.processor.filter_blocks_by_ratio(blocks, 0.5)) == 5
        assert len(self.processor.filter_blocks_by_ratio(blocks, 0.1)) == 1
        assert len(self.processor.filter_blocks_by_ratio(blocks, 0.0)) == 0
    
    def test_validate_text_quality(self):
        """Test text quality validation."""
        # Good quality text
        good_text = "这是一个包含中文的高质量文本，内容丰富有意义。"
        assert self.processor.validate_text_quality(good_text)
        
        # Poor quality text
        poor_cases = [
            "",  # Empty
            "   ",  # Only whitespace
            "abc123",  # Too short
            "1234567890" * 10,  # No Chinese characters
        ]
        
        for text in poor_cases:
            assert not self.processor.validate_text_quality(text), f"Failed for: {text}"


class TestTextProcessorCustomPrefixes:
    """Test TextProcessor with custom prefixes."""
    
    def test_custom_prefixes(self):
        """Test text processor with custom known prefixes."""
        custom_prefixes = ["测试前缀", "自定义"]
        processor = TextProcessor(known_prefixes=custom_prefixes)
        
        # Test cleaning with custom prefix
        result = processor.clean_question_text("测试前缀：这是问题")
        assert result == "这是问题"
        
        # Test that default prefixes still work (fallback pattern)
        result = processor.clean_question_text("网友：这也是问题")
        assert result == "这也是问题" 