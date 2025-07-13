"""Unit tests for enhanced QA Extractor features."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.core.qa_extractor import QAExtractor


class TestQAExtractorEnhanced:
    """Test cases for enhanced QA Extractor features."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'long_answer_processing': {
                'chain_summary_threshold': 100,  # Low threshold for testing
                'summary_length': 20,
                'nli_model_path': 'test-model',
                'entailment_threshold': 0.7
            }
        }
    
    @pytest.fixture
    def qa_extractor(self, config):
        """Create QAExtractor instance."""
        return QAExtractor(max_prompt_tokens=1000, config=config)
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        client.call_ollama = Mock(return_value="这是一个测试摘要")
        return client
    
    def test_init_with_config(self, qa_extractor):
        """Test QAExtractor initialization with config."""
        assert qa_extractor.chain_summary_threshold == 100
        assert qa_extractor.summary_length == 20
        assert qa_extractor.entailment_threshold == 0.7
        assert qa_extractor.nli_model_path == 'test-model'
    
    def test_extract_key_sentences(self, qa_extractor):
        """Test extractive summarization."""
        text = "第一句话。第二句话。第三句话。第四句话。第五句话。"
        
        # Extract 3 sentences
        result = qa_extractor._extract_key_sentences(text, num_sentences=3)
        
        # Should contain first, middle, and last sentences
        assert "第一句话" in result
        assert "第五句话" in result
        assert len(result.split("。")) >= 3
    
    def test_generate_summary_with_llm(self, qa_extractor, mock_llm_client):
        """Test LLM summary generation."""
        text = "这是一段很长的文本" * 50
        
        summary = qa_extractor._generate_summary_with_llm(text, mock_llm_client)
        
        # Should call LLM
        mock_llm_client.call_ollama.assert_called_once()
        
        # Should return cleaned summary
        assert summary == "这是一个测试摘要"
        assert len(summary) <= qa_extractor.summary_length
    
    @patch('src.core.qa_extractor.AutoTokenizer')
    @patch('src.core.qa_extractor.AutoModelForSequenceClassification')
    def test_check_entailment(self, mock_model_class, mock_tokenizer_class, qa_extractor):
        """Test entailment checking."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {'input_ids': [1, 2, 3]}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_logits = MagicMock()
        mock_logits.logits = MagicMock()
        
        # Create tensor-like object for probabilities
        mock_probs = MagicMock()
        mock_probs.__getitem__ = lambda self, idx: MagicMock(
            __getitem__=lambda self2, idx2: MagicMock(item=lambda: 0.8)
        )
        
        # Mock softmax to return our mock probabilities
        with patch('torch.softmax', return_value=mock_probs):
            mock_model.return_value = mock_logits
            mock_model.eval = Mock()
            mock_model_class.from_pretrained.return_value = mock_model
            
            # Test entailment check
            premise = "这是前提"
            hypothesis = "这是假设"
            
            is_entailed, score = qa_extractor._check_entailment(premise, hypothesis)
            
            # Should return True for score > threshold
            assert is_entailed is True
            assert score == 0.8
    
    def test_process_long_answer_short(self, qa_extractor, mock_llm_client):
        """Test that short answers are not processed."""
        short_answer = "这是一个短答案"
        
        result = qa_extractor._process_long_answer(short_answer, mock_llm_client)
        
        # Should return original answer
        assert result == short_answer
        
        # Should not call LLM
        mock_llm_client.call_ollama.assert_not_called()
    
    @patch.object(QAExtractor, '_check_entailment')
    def test_process_long_answer_with_chain_summary(self, mock_check_entailment, qa_extractor, mock_llm_client):
        """Test long answer processing with chain summarization."""
        # Create a long answer
        long_answer = "这是一段很长的答案。" * 50  # About 400 chars
        
        # Mock entailment check to return True
        mock_check_entailment.return_value = (True, 0.8)
        
        # Mock LLM to return different summaries
        mock_llm_client.call_ollama.side_effect = ["第一部分摘要", "第二部分摘要", "第三部分摘要", "最后部分"]
        
        result = qa_extractor._process_long_answer(long_answer, mock_llm_client)
        
        # Should call LLM multiple times
        assert mock_llm_client.call_ollama.call_count >= 2
        
        # Result should be shorter than original
        assert len(result) < len(long_answer)
        
        # Should contain summaries
        assert "摘要" in result
    
    def test_extract_from_high_confidence_block(self, qa_extractor):
        """Test rule-based extraction from high confidence blocks."""
        content = """网友：什么是价值投资？
段永平：价值投资就是买便宜的好公司。
要看公司的基本面。

问：如何判断公司价值？
答：主要看现金流和护城河。"""
        
        qa_pairs = qa_extractor._extract_from_high_confidence_block(content)
        
        # Should extract 2 Q&A pairs
        assert len(qa_pairs) == 2
        
        # Check first pair
        assert qa_pairs[0]['question'] == "什么是价值投资？"
        assert "买便宜的好公司" in qa_pairs[0]['answer']
        
        # Check second pair
        assert qa_pairs[1]['question'] == "如何判断公司价值？"
        assert "现金流" in qa_pairs[1]['answer']
    
    def test_process_groups(self, qa_extractor, mock_llm_client):
        """Test processing semantic groups."""
        groups = [
            {
                'content': "网友：测试问题？\n段：测试答案。",
                'confidence': 'high',
                'type': 'rule_based',
                'domain': 'general'
            },
            {
                'content': "这是一个需要LLM处理的段落",
                'confidence': 'low',
                'type': 'semantic',
                'domain': 'general'
            }
        ]
        
        # Mock LLM extraction
        with patch.object(qa_extractor, '_extract_with_llm') as mock_extract:
            mock_extract.return_value = [{'question': 'LLM问题', 'answer': 'LLM答案'}]
            
            result = qa_extractor.process_groups(groups, mock_llm_client)
            
            # Should process both groups
            assert len(result) >= 1
            
            # High confidence should use rule-based extraction
            high_conf_pairs = [p for p in result if p.get('source_confidence') == 'high']
            assert len(high_conf_pairs) > 0
            
            # All pairs should have metadata
            for pair in result:
                assert 'source_confidence' in pair
                assert 'source_type' in pair
                assert 'domain' in pair
    
    def test_error_handling_in_process_groups(self, qa_extractor, mock_llm_client):
        """Test error handling in process_groups."""
        groups = [
            {
                'content': None,  # Invalid content
                'confidence': 'high'
            }
        ]
        
        # Should not raise exception
        result = qa_extractor.process_groups(groups, mock_llm_client)
        
        # Should return empty list or skip invalid group
        assert isinstance(result, list) 