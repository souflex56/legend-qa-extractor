"""Q&A extraction module for processing and extracting question-answer pairs."""

import json
import re
from typing import List, Dict, Any, Optional
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..config import Config

logger = logging.getLogger(__name__)


class QAExtractor:
    """Handles extraction and processing of Q&A pairs from LLM responses."""
    
    def __init__(self, llm_client, max_prompt_tokens: int = 6000, config: Optional[Config] = None):
        self.config = config if config else Config()
        self.llm_client = llm_client
        self.logger = logger
        self.max_prompt_tokens = max_prompt_tokens

        # 加载一个NLI模型用于蕴含校验
        nli_model_path = getattr(self.config, "nli_model_path", "cross-encoder/nli-deberta-v3-base")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_path)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_path)
        
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
    
    def _check_entailment(self, premise: str, hypothesis: str) -> bool:
        # 蕴含校验核心逻辑
        input_pair = [[premise, hypothesis]]
        features = self.nli_tokenizer(input_pair, padding=True, truncation=True, return_tensors="pt")
        self.nli_model.eval()
        with torch.no_grad():
            scores = self.nli_model(**features).logits
            # label_mapping = ['contradiction', 'entailment', 'neutral']
            # 我们只关心蕴含（entailment）的概率
            entailment_score = torch.softmax(scores, dim=1)[0][1].item() 
        
        return entailment_score > getattr(self.config, "entailment_threshold", 0.7)

    def _process_long_answer(self, group_text: str) -> List[Dict[str, Any]]:
        # 链式摘要处理逻辑
        # ... 实现切分、调用LLM生成摘要、调用_check_entailment进行校验
        # ... 如果校验失败，可以采用抽取式摘要或更严格的LLM Prompt作为fallback
        # For now, we just call the single block processor
        return self._process_single_block(group_text)
    
    def process_groups(self, groups: list[dict]):
        """Process multiple groups with progress tracking and error handling."""
        final_qa_pairs = []
        total_groups = len(groups)
        successful_groups = 0
        failed_groups = 0
        
        self.logger.info(f"🚀 Starting processing {total_groups} groups...")
        
        for i, group in enumerate(groups, 1):
            self.logger.info(f"🔄 Processing group {i}/{total_groups} ({len(group['content'])} chars)...")
            
            try:
                group_text = group['content']
                chain_summary_threshold = getattr(self.config, "chain_summary_threshold", 3000)
                
                if len(group_text) > chain_summary_threshold:
                    self.logger.info(f"📄 Group {i} is long ({len(group_text)} chars), using chain summary...")
                    qa_pairs = self._process_long_answer(group_text)
                else:
                    self.logger.info(f"📄 Group {i} is normal size, using single block processing...")
                    qa_pairs = self._process_single_block(group_text)
                
                if qa_pairs:
                    final_qa_pairs.extend(qa_pairs)
                    successful_groups += 1
                    self.logger.info(f"✅ Group {i} processed successfully: {len(qa_pairs)} Q&A pairs extracted")
                else:
                    failed_groups += 1
                    self.logger.warning(f"⚠️ Group {i} processed but no Q&A pairs extracted")
                    
            except Exception as e:
                failed_groups += 1
                self.logger.error(f"❌ Group {i} processing failed: {e}")
                self.logger.debug(f"Failed group content preview: {group_text[:200]}...")
        
        self.logger.info(f"🎉 Processing completed:")
        self.logger.info(f"  ✅ Successful groups: {successful_groups}/{total_groups}")
        self.logger.info(f"  ❌ Failed groups: {failed_groups}/{total_groups}")
        self.logger.info(f"  🎯 Total Q&A pairs extracted: {len(final_qa_pairs)}")
        
        return final_qa_pairs

    def _process_single_block(self, block_text: str) -> Optional[List[Dict[str, str]]]:
        """Process a single block of text to extract Q&A pairs."""
        try:
            # Choose prompt based on available token space
            prompt_template = self.full_prompt
            if self.estimate_token_count(block_text) > (self.max_prompt_tokens - 500):
                prompt_template = self.compact_prompt
            
            prompt = f"{prompt_template}\n\n{block_text}"
            
            # Smart truncation if still too long
            # This is a fallback and should ideally not be hit if blocks are sized well
            if self.estimate_token_count(prompt) > self.max_prompt_tokens:
                # Calculate allowed chars for the block text
                allowed_chars = int(len(block_text) * (self.max_prompt_tokens / self.estimate_token_count(prompt)))
                block_text = self._smart_truncate_text(block_text, allowed_chars)
                prompt = f"{prompt_template}\n\n{block_text}"

            # Call LLM
            response_text = self.llm_client.call_ollama(prompt, temperature=self.config.temperature)
            
            if not response_text:
                self.logger.warning(f"No response from LLM for block: {block_text[:100]}...")
                return None

            self.logger.debug(f"LLM raw response:\n{response_text}")
            
            # Extract JSON from response
            qa_pairs = self.extract_json(response_text)
            
            if qa_pairs:
                self.logger.info(f"Successfully extracted {len(qa_pairs)} Q&A pairs from block.")
                return qa_pairs
            else:
                self.logger.warning(f"No Q&A pairs extracted from block: {block_text[:100]}...")
                return None

        except Exception as e:
            self.logger.error(f"Error processing block: {e}")
            self.logger.debug(f"Block content that caused error: {block_text[:500]}")
            return None

    def estimate_token_count(self, text: str) -> int:
        """估算文本的token数量（中文约1.5倍字符数）"""
        # 中文字符和token比例约1:1.5，英文约1:0.75，取保守估计
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.75)
    
    def get_full_prompt(self, text_block: str, block_anchor: str = "", sliding_context: str = "") -> str:
        """Creates a full prompt for the LLM, managing token limits."""
        # This is a simplified version; in a real scenario, you'd have more logic
        # to decide between compact and full prompts, and to truncate text.
        return f"{self.full_prompt}\n\nContext: {sliding_context}\nAnchor: {block_anchor}\n\n{text_block}"

    def validate_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validates a list of QA pairs."""
        # Simple validation, can be expanded
        return [p for p in qa_pairs if "question" in p and "answer" in p]

    def generate_topic(self, question: str, answer: str) -> str:
        """Generates a topic for a Q&A pair."""
        # In a real implementation, this would call the LLM
        prompt = f"Generate a short, concise topic for the following Q&A pair:\n\nQuestion: {question}\nAnswer: {answer}\n\nTopic:"
        # response = self.llm_client.generate(prompt)
        # return response.strip()
        return "Generated Topic" # Placeholder

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
            json_blocks = re.findall(r'```json\\s*(.*?)\\s*```', text, re.DOTALL)
            
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
        """Extract multiple JSON objects from a single string.
        
        Args:
            content: String possibly containing multiple JSON objects
            
        Returns:
            List of JSON object strings
        """
        json_objects = []
        brace_level = 0
        start_index = -1
        
        for i, char in enumerate(content):
            if char == '{':
                if brace_level == 0:
                    start_index = i
                brace_level += 1
            elif char == '}':
                brace_level -= 1
                if brace_level == 0 and start_index != -1:
                    json_objects.append(content[start_index:i+1])
                    start_index = -1
                    
        return json_objects
    
    def _is_valid_qa_pair(self, data: Any) -> bool:
        """Check if a dictionary is a valid Q&A pair."""
        return (isinstance(data, dict) and 
                "question" in data and "answer" in data and
                isinstance(data["question"], str) and data["question"].strip() and
                isinstance(data["answer"], str) and data["answer"].strip())
    
    def process_qa_pairs(self, qa_pairs: List[Dict[str, Any]], 
                        source_text: str, 
                        text_processor) -> List[Dict[str, Any]]:
        """Process extracted Q&A pairs for quality and formatting.
        
        Args:
            qa_pairs: List of extracted Q&A pairs
            source_text: The original text block the pairs were extracted from
            text_processor: The text processor for cleaning
            
        Returns:
            List of processed and cleaned Q&A pairs
        """
        processed_pairs = []
        for pair in qa_pairs:
            # Clean up text
            pair["question"] = text_processor.clean_question_text(pair["question"])
            pair["answer"] = pair["answer"]
            
            # Add source text metadata
            pair["source_text"] = source_text
            
            processed_pairs.append(pair)
        
        return processed_pairs
    
    def validate_extraction_quality(self, original_text: str, 
                                   qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of extracted Q&A pairs against the source text.
        
        Args:
            original_text: The source text
            qa_pairs: List of extracted Q&A pairs
            
        Returns:
            A dictionary with validation status and feedback.
        """
        if not qa_pairs:
            return { "is_valid": True, "feedback": "No Q&A pairs to validate." }
        
        is_valid = True
        feedback = []
        
        # Example validation: Check if question and answer text appear in the source
        for i, pair in enumerate(qa_pairs):
            q_text = pair['question'].split('：', 1)[-1].strip()
            a_text = pair['answer'].split('：', 1)[-1].strip()
            
            if q_text not in original_text:
                is_valid = False
                feedback.append(f"Pair {i+1}: Question text not found in source.")
            
            if a_text not in original_text:
                is_valid = False
                feedback.append(f"Pair {i+1}: Answer text not found in source.")
        
        return {
            "is_valid": is_valid,
            "feedback": feedback
        }