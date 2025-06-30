"""Q&A extraction module for processing and extracting question-answer pairs."""

import json
import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class QAExtractor:
    """Handles extraction and processing of Q&A pairs from LLM responses."""
    
    def __init__(self):
        self.logger = logger
        self.base_prompt = self._get_base_prompt()
    
    def _get_base_prompt(self) -> str:
        """Get the base prompt for Q&A extraction."""
        return """
你是一个专业的中文问答对提取专家。请从给定的原文中提取**所有**有效的问答对。

🎯 核心任务：
1. 识别所有形式的提问或话题引入
2. 匹配对应的段永平回答
3. 确保问答配对准确完整

📋 提取规则：
1. **问题来源**（多种形式）：
   - 直接提问：网友、问、观众、主持人、Q等开头
   - 文章引用：文章引用、引用、有人说等引出的观点或问题
   - 间接提问：通过描述、举例引出的问题
   - 话题讨论：任何引发段永平回应的内容

2. **答案来源**：仅限段永平、段、大道的回答（排除方丈、其他人）

3. **配对原则**：
   - 一个问题/话题对应一个段永平的回答
   - 问题可以是直接提问，也可以是引用、描述等形式
   - 保持问题和答案的完整性和上下文

4. **输出格式**：JSON数组 [{"question": "问题或话题", "answer": "段永平的回答"}]

❌ 不要提取的内容：
- 方丈的回答
- 其他专家、学者的观点（除非段永平对此有回应）
- 纯描述性文字（除非段永平对此有回应）
- 目录、标题等

✅ 提取示例：

原文1（直接提问）：
网友：什么是stop doing list？
段永平：所谓要做对的事情实际上是通过不做不对的事情来实现的。

原文2（文章引用形式）：
文章引用："微信之父"张小龙就曾说，乔布斯最厉害的地方是他1秒钟就能变成傻瓜。
段：是不是张小龙说的不知道，但这话其实很有道理。

原文3（描述引出）：
有人认为价值投资很难学会。
段：价值投资确实不容易，但有悟性的人可以提高。

输出：
[
  {"question": "什么是stop doing list？", "answer": "所谓要做对的事情实际上是通过不做不对的事情来实现的。"},
  {"question": "文章引用：微信之父张小龙就曾说，乔布斯最厉害的地方是他1秒钟就能变成傻瓜。", "answer": "是不是张小龙说的不知道，但这话其实很有道理。"},
  {"question": "有人认为价值投资很难学会。", "answer": "价值投资确实不容易，但有悟性的人可以提高。"}
]

🔍 请仔细分析以下原文，识别所有引发段永平回应的内容（包括直接提问、文章引用、描述等），并提取所有符合条件的问答对：
"""
    
    def create_prompt(self, text: str) -> str:
        """Create complete prompt with the given text.
        
        Args:
            text: Text to extract Q&A pairs from
            
        Returns:
            Complete prompt string
        """
        return self.base_prompt + "\n\n" + text
    
    def extract_json(self, text: str) -> List[Dict[str, Any]]:
        """Extract JSON data from LLM response.
        
        Args:
            text: LLM response text containing JSON
            
        Returns:
            List of Q&A pair dictionaries
        """
        results = []
        
        try:
            # Try to find JSON blocks wrapped in ```json```
            json_blocks = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            
            if json_blocks:
                for json_block in json_blocks:
                    results.extend(self._parse_json_content(json_block))
            else:
                # If no ```json``` wrapper, parse the entire text
                results.extend(self._parse_json_content(text))
        
        except Exception as e:
            self.logger.error(f"JSON extraction error: {e}\nOriginal response:\n{text}")
        
        # Filter valid Q&A pairs
        valid_results = []
        for data in results:
            if self._is_valid_qa_pair(data):
                valid_results.append(data)
        
        return valid_results
    
    def _parse_json_content(self, content: str) -> List[Dict[str, Any]]:
        """Parse JSON content, handling arrays, single objects, and multiple objects.
        
        Args:
            content: JSON content string to parse
            
        Returns:
            List of parsed JSON objects
        """
        results = []
        content = content.strip()
        
        if not content:
            return results
        
        try:
            # First try to parse as JSON (could be array or single object)
            data = json.loads(content)
            if isinstance(data, list):
                # If it's an array, extend results
                for item in data:
                    if isinstance(item, dict):
                        results.append(item)
            elif isinstance(data, dict):
                # If it's a single object, add to results
                results.append(data)
            return results
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to separate multiple JSON objects
            try:
                json_objects = self._extract_json_objects(content)
                for json_str in json_objects:
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, dict):
                            results.append(data)
                    except json.JSONDecodeError:
                        continue
                        
            except Exception as e:
                self.logger.error(f"JSON object separation failed: {e}\nContent:\n{content}")
        
        return results
    
    def _extract_json_objects(self, content: str) -> List[str]:
        """Extract individual JSON objects from text containing multiple objects.
        
        Args:
            content: Text content containing JSON objects
            
        Returns:
            List of JSON object strings
        """
        json_objects = []
        brace_count = 0
        start_pos = -1
        
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    json_str = content[start_pos:i+1]
                    json_objects.append(json_str)
                    start_pos = -1
        
        return json_objects
    
    def _is_valid_qa_pair(self, data: Any) -> bool:
        """Check if the data is a valid Q&A pair.
        
        Args:
            data: Data object to validate
            
        Returns:
            True if valid Q&A pair, False otherwise
        """
        return (isinstance(data, dict) and 
                "question" in data and "answer" in data and
                data.get("question") and data.get("answer") and
                str(data.get("question", "")).strip() and 
                str(data.get("answer", "")).strip())
    
    def process_qa_pairs(self, qa_pairs: List[Dict[str, Any]], 
                        source_text: str, 
                        text_processor) -> List[Dict[str, Any]]:
        """Process extracted Q&A pairs with cleaning and formatting.
        
        Args:
            qa_pairs: List of raw Q&A pairs
            source_text: Original source text
            text_processor: TextProcessor instance for cleaning
            
        Returns:
            List of processed Q&A pairs
        """
        processed_pairs = []
        
        for qa_pair in qa_pairs:
            if not self._is_valid_qa_pair(qa_pair):
                continue
            
            # Clean question text
            clean_question = text_processor.clean_question_text(qa_pair["question"])
            
            # Create final Q&A pair
            final_pair = {
                "question": clean_question,
                "answer": qa_pair["answer"],
                "source_text": source_text
            }
            
            processed_pairs.append(final_pair)
        
        return processed_pairs
    
    def validate_extraction_quality(self, original_text: str, 
                                   qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of extracted Q&A pairs.
        
        Args:
            original_text: Original text that was processed
            qa_pairs: Extracted Q&A pairs
            
        Returns:
            Quality metrics dictionary
        """
        metrics = {
            'total_pairs': len(qa_pairs),
            'avg_question_length': 0,
            'avg_answer_length': 0,
            'has_duplicates': False,
            'quality_score': 0.0
        }
        
        if not qa_pairs:
            return metrics
        
        # Calculate average lengths
        question_lengths = [len(pair['question']) for pair in qa_pairs]
        answer_lengths = [len(pair['answer']) for pair in qa_pairs]
        
        metrics['avg_question_length'] = sum(question_lengths) / len(question_lengths)
        metrics['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)
        
        # Check for duplicates
        questions = [pair['question'] for pair in qa_pairs]
        metrics['has_duplicates'] = len(questions) != len(set(questions))
        
        # Calculate quality score (simple heuristic)
        score = 0.0
        if metrics['avg_question_length'] > 5:
            score += 0.3
        if metrics['avg_answer_length'] > 10:
            score += 0.3
        if not metrics['has_duplicates']:
            score += 0.2
        if metrics['total_pairs'] > 0:
            score += 0.2
        
        metrics['quality_score'] = score
        
        return metrics 