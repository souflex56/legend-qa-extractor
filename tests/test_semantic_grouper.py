"""Unit tests for Semantic Grouper."""

import pytest
import numpy as np
from src.core.semantic_grouper import SemanticGrouper


class TestSemanticGrouper:
    """Test cases for SemanticGrouper class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'semantic_grouping': {
                'max_question_length': 50,
                'default_similarity_threshold': 0.65,
                'std_factor': 0.5
            },
            'max_block_size': 1500,
            'min_block_size': 200
        }
    
    @pytest.fixture
    def grouper(self, config):
        """Create SemanticGrouper instance."""
        return SemanticGrouper(config)
    
    def test_init(self, grouper):
        """Test SemanticGrouper initialization."""
        assert grouper.max_question_length == 50
        assert grouper.default_similarity_threshold == 0.65
        assert grouper.std_factor == 0.5
        assert 'general' in grouper.models
    
    def test_detect_domain(self, grouper):
        """Test domain detection."""
        # Test financial domain
        financial_text = "这是一份财报，显示公司市值增长了20%，投资收益率达到15%"
        assert grouper._detect_domain(financial_text) == "financial"
        
        # Test medical domain
        medical_text = "患者服用该药品后，症状明显改善，医生建议继续治疗"
        assert grouper._detect_domain(medical_text) == "medical"
        
        # Test general domain
        general_text = "今天天气很好，适合出去散步"
        assert grouper._detect_domain(general_text) == "general"
    
    def test_is_potential_question(self, grouper):
        """Test potential question detection."""
        # Test questions
        assert grouper._is_potential_question("什么是价值投资？")
        assert grouper._is_potential_question("为什么要这样做？")
        assert grouper._is_potential_question("如何提高投资回报率")
        
        # Test non-questions
        assert not grouper._is_potential_question("这是一个很长的段落，包含了很多内容，超过了50个字符的限制")
        assert not grouper._is_potential_question("今天天气很好。")
    
    def test_calculate_dynamic_threshold(self, grouper):
        """Test dynamic threshold calculation."""
        # Test with sufficient paragraphs
        paragraphs = [
            "投资的核心是什么",
            "投资的核心是寻找价值",
            "价值投资需要耐心"
        ]
        
        model = grouper.models['general']
        threshold = grouper._calculate_dynamic_threshold(paragraphs, model)
        
        # Should be within valid range
        assert 0.5 <= threshold <= 0.9
        
        # Test with insufficient paragraphs
        threshold = grouper._calculate_dynamic_threshold(["单个段落"], model)
        assert threshold == grouper.default_similarity_threshold
    
    def test_rule_based_prescreening(self, grouper):
        """Test rule-based prescreening."""
        paragraphs = [
            "网友：什么是价值投资？",
            "段永平：价值投资就是买便宜的好公司。",
            "这是一个普通段落。",
            "问：如何选择股票？",
            "答：要看公司的基本面。"
        ]
        
        high_confidence_blocks, remaining_indices = grouper._rule_based_prescreening(paragraphs)
        
        # Should find at least one high confidence block
        assert len(high_confidence_blocks) > 0
        
        # Check block content
        first_block = high_confidence_blocks[0]
        assert '网友' in first_block['content']
        assert first_block['confidence'] == 'high'
        assert first_block['type'] == 'rule_based'
        
        # Check remaining indices
        assert 2 in remaining_indices  # "这是一个普通段落。"
    
    def test_merge_and_optimize_blocks(self, grouper):
        """Test block merging and optimization."""
        blocks = [
            {
                'content': 'A' * 100,  # Small block
                'confidence': 'high',
                'type': 'rule_based',
                'indices': [0]
            },
            {
                'content': 'B' * 150,  # Small block
                'confidence': 'medium',
                'type': 'semantic',
                'indices': [1]
            },
            {
                'content': 'C' * 1000,  # Normal block
                'confidence': 'high',
                'type': 'rule_based',
                'indices': [2]
            }
        ]
        
        optimized = grouper._merge_and_optimize_blocks(blocks)
        
        # Small blocks should be merged
        assert len(optimized) <= len(blocks)
        
        # All blocks should be within size limits
        for block in optimized:
            assert len(block['content']) >= grouper.min_block_size
            assert len(block['content']) <= grouper.max_block_size
    
    def test_split_large_block(self, grouper):
        """Test large block splitting."""
        large_block = {
            'content': 'A' * 2000 + '\n\n' + 'B' * 2000,  # Oversized block
            'confidence': 'high',
            'type': 'rule_based',
            'indices': [0, 1]
        }
        
        split_blocks = grouper._split_large_block(large_block)
        
        # Should be split into multiple blocks
        assert len(split_blocks) > 1
        
        # Each block should be within size limit
        for block in split_blocks:
            assert len(block['content']) <= grouper.max_block_size
    
    def test_group_integration(self, grouper):
        """Test complete grouping process."""
        paragraphs = [
            "网友：什么是价值投资？",
            "段永平：价值投资就是买便宜的好公司。",
            "要看公司的护城河。",
            "还要看管理层。",
            "这是一个独立的段落。",
            "问：如何判断公司价值？",
            "要看财报和现金流。"
        ]
        
        result = grouper.group(paragraphs)
        
        # Should produce reasonable number of blocks
        assert len(result) > 0
        assert len(result) <= len(paragraphs)
        
        # Each block should have required fields
        for block in result:
            assert 'content' in block
            assert 'confidence' in block
            assert 'type' in block
            
        # Check confidence distribution is logged
        # (This would be checked through log inspection in real tests) 