"""Q&A extraction module for processing and extracting question-answer pairs."""

import json
import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class QAExtractor:
    """Handles extraction and processing of Q&A pairs from LLM responses."""
    
    def __init__(self, max_prompt_tokens: int = 6000):
        self.logger = logger
        self.max_prompt_tokens = max_prompt_tokens  # 留一些余量给模型
        
        # 精简版基础prompt，保留核心功能但大幅缩短
        self.compact_prompt = """你是中文问答对提取专家，从原文提取段永平的所有真实问答对。

【核心原则】必须有真实外部提问（网友/主持人/引用观点）引发段永平回应，每个问题对应一个完整回答（含连续补充），禁止提取修辞性自问句。

【提取流程】找到真实外部问题 → 匹配完整段永平回答 → 严格验证配对

【严禁】段永平阐述中的"什么是XX？""很难吗？"等修辞问句、问答内容相同/颠倒、无外部引发就输出

输出格式：JSON数组 [{"question": "外部问题", "answer": "段永平完整回答"}]

原文："""
        
        # 完整版基础prompt（当有充足空间时使用）
        self.full_prompt = """你是专业中文问答对提取专家，从原文提取段永平的所有有效问答对。

🎯 【核心原则】
• 必须存在真实外部提问（网友、主持人、引用观点等）引发段永平回应
• 每个外部问题只对应一个完整合并回答（包含所有后续补充片段）
• 绝对禁止提取段永平阐述中的修辞性自问句

📋 【提取流程】
1️⃣ **问题识别**：明确标识（网友：、问：）或引用观点（有人说、文章引用）
2️⃣ **回答匹配**：段永平的完整连续回应（含所有相关补充）
3️⃣ **配对验证**：问题与回答逻辑对应，无内容重复/颠倒

🔧 【边界处理】
• **同一问题的离散回答**：合并为一个完整answer
• **新问题判断**：出现新提问者或话题实质转换
• **修辞性问句**：段永平论述中的"什么是XX？""很难吗？"等不提取

✅ **核心示例**
```
网友：什么是stop doing list？
段永平：所谓要做对的事情实际上是通过不做不对的事情来实现的。

有人认为价值投资已经过时了。
我不这么认为。价值投资永远不会过时，因为它的本质是买优秀的公司。

主持人：投资中最难的是什么？
段永平：最难的是克服恐惧和贪婪。这是人性。
主持人：还有吗？
段永平：还有就是坚持不懂不做。只在自己的能力圈内活动。
```

正确输出：
```json
[
  {"question": "什么是stop doing list？", "answer": "所谓要做对的事情实际上是通过不做不对的事情来实现的。"},
  {"question": "有人认为价值投资已经过时了。", "answer": "我不这么认为。价值投资永远不会过时，因为它的本质是买优秀的公司。"},
  {"question": "投资中最难的是什么？", "answer": "最难的是克服恐惧和贪婪。这是人性。还有就是坚持不懂不做。只在自己的能力圈内活动。"}
]
```

❌ **集中错误防范**
• 把段永平的修辞问句当外部问题（如"价值投资的核心是什么？就是买优秀公司"中的问句）
• 问题和答案内容相同或逻辑颠倒
• 拆分属于同一问题的连续回答片段
• 无外部问题引发就输出问答对

🔍 请仔细分析以下原文，提取所有符合条件的问答对：
"""
    
    def estimate_token_count(self, text: str) -> int:
        """估算文本的token数量（中文约1.5倍字符数）"""
        # 中文字符和token比例约1:1.5，英文约1:0.75，取保守估计
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.75)
    
    def create_prompt(self, text_block: str, sliding_context: str = "", block_anchor: str = "") -> str:
        """创建完整的LLM提示，支持智能长度管理。
        
        Args:
            text_block: 需要提取问答对的文本内容
            sliding_context: 前一个相关文本块的末尾部分，用于提供额外上下文
            block_anchor: 当前文本块的核心主题或关键词
            
        Returns:
            完整的提示文本
        """
        # 构建上下文部分
        context_section = ""
        if sliding_context:
            context_section += f"\n\n上下文：{sliding_context[:200]}...\n"  # 限制上下文长度
        
        if block_anchor:
            context_section += f"\n主题：{block_anchor}\n"
        
        # 尝试使用完整prompt
        full_prompt_text = f"{self.full_prompt}{context_section}\n\n{text_block}"
        full_tokens = self.estimate_token_count(full_prompt_text)
        
        # 如果完整prompt不超限，使用完整版
        if full_tokens <= self.max_prompt_tokens:
            self.logger.debug(f"Using full prompt, estimated tokens: {full_tokens}")
            return full_prompt_text
        
        # 否则使用精简版prompt
        compact_prompt_text = f"{self.compact_prompt}{context_section}\n\n{text_block}"
        compact_tokens = self.estimate_token_count(compact_prompt_text)
        
        # 如果精简版仍超限，需要截断文本块
        if compact_tokens > self.max_prompt_tokens:
            # 计算可用于文本块的token数
            base_tokens = self.estimate_token_count(f"{self.compact_prompt}{context_section}")
            available_tokens = self.max_prompt_tokens - base_tokens - 100  # 留100token余量
            
            # 估算可容纳的字符数
            available_chars = int(available_tokens / 1.5)  # 保守估计
            
            if available_chars > 100:  # 确保有最小的文本量
                truncated_block = self._smart_truncate_text(text_block, available_chars)
                compact_prompt_text = f"{self.compact_prompt}{context_section}\n\n{truncated_block}"
                self.logger.warning(f"Text block truncated to {len(truncated_block)} chars due to token limit")
            else:
                # 如果连最小文本都放不下，去掉上下文信息
                compact_prompt_text = f"{self.compact_prompt}\n\n{text_block[:available_chars]}"
                self.logger.warning(f"Context removed due to token limit, text truncated to {available_chars} chars")
        
        final_tokens = self.estimate_token_count(compact_prompt_text)
        self.logger.debug(f"Using compact prompt, estimated tokens: {final_tokens}")
        return compact_prompt_text
    
    def _smart_truncate_text(self, text: str, max_chars: int) -> str:
        """智能截断文本，尽量保持完整性"""
        if len(text) <= max_chars:
            return text
        
        # 尝试按段落截断
        paragraphs = text.split('\n\n')
        result = ""
        for para in paragraphs:
            if len(result) + len(para) + 2 <= max_chars:  # +2 for \n\n
                result += para + "\n\n" if result else para
            else:
                break
        
        # 如果按段落截断后太短，尝试按句子截断
        if len(result) < max_chars * 0.7:  # 如果截断后少于70%，尝试句子级截断
            sentences = re.split(r'(?<=[。！？；])', text)
            result = ""
            for sentence in sentences:
                if len(result) + len(sentence) <= max_chars:
                    result += sentence
                else:
                    break
        
        # 最后确保不超长
        if len(result) > max_chars:
            result = result[:max_chars-3] + "..."
        
        return result.strip()
    
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