"""Q&A extraction module for processing and extracting question-answer pairs."""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class QAExtractor:
    """Handles extraction and processing of Q&A pairs from LLM responses."""
    
    def __init__(self, max_prompt_tokens: int = 6000, config: Optional[Dict] = None):
        self.logger = logger
        self.max_prompt_tokens = max_prompt_tokens  # 留一些余量给模型
        self.config = config or {}
        
        # 长答案处理配置
        long_answer_config = self.config.get('long_answer_processing', {})
        self.chain_summary_threshold = long_answer_config.get('chain_summary_threshold', 3000)
        self.summary_length = long_answer_config.get('summary_length', 50)
        self.entailment_threshold = long_answer_config.get('entailment_threshold', 0.7)
        
        # 初始化NLI模型（延迟加载）
        self.nli_model = None
        self.nli_tokenizer = None
        self.nli_model_path = long_answer_config.get('nli_model_path', 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')
        
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
    
    def _load_nli_model(self):
        """延迟加载NLI模型，只在需要时加载"""
        if self.nli_model is None:
            self.logger.info(f"Loading NLI model: {self.nli_model_path}")
            try:
                self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_path)
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_path)
                self.nli_model.eval()
                self.logger.info("NLI model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load NLI model: {e}")
                raise
    
    def _check_entailment(self, premise: str, hypothesis: str) -> Tuple[bool, float]:
        """
        蕴含校验：检查假设(hypothesis)是否被前提(premise)所蕴含
        返回：(是否蕴含, 蕴含分数)
        """
        self._load_nli_model()
        
        try:
            # 准备输入
            inputs = self.nli_tokenizer(
                premise, 
                hypothesis, 
                truncation=True, 
                padding=True, 
                max_length=512,
                return_tensors="pt"
            )
            
            # 推理
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                logits = outputs.logits
                
                # 获取概率分布
                probabilities = torch.softmax(logits, dim=-1)
                
                # mDeBERTa-v3-base-xnli 的标签映射：
                # 0: entailment, 1: neutral, 2: contradiction
                entailment_score = probabilities[0][0].item()
                
            is_entailed = entailment_score >= self.entailment_threshold
            
            self.logger.debug(f"Entailment check - Score: {entailment_score:.3f}, Threshold: {self.entailment_threshold}")
            
            return is_entailed, entailment_score
            
        except Exception as e:
            self.logger.error(f"Error in entailment check: {e}")
            # 出错时返回保守结果
            return False, 0.0
    
    def _generate_summary_with_llm(self, text: str, llm_client) -> str:
        """使用LLM生成文本摘要"""
        prompt = f"""请将以下内容总结为{self.summary_length}字以内的摘要，保留核心信息：

{text[:2000]}  # 限制输入长度

摘要："""
        
        try:
            summary = llm_client.call_ollama(prompt, temperature=0.1)
            if summary:
                # 清理摘要
                summary = summary.strip().replace("摘要：", "").replace("总结：", "").strip()
                # 确保不超过指定长度
                if len(summary) > self.summary_length:
                    summary = summary[:self.summary_length-3] + "..."
                return summary
            else:
                return ""
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return ""
    
    def _extract_key_sentences(self, text: str, num_sentences: int = 3) -> str:
        """抽取式摘要：提取关键句子作为备用方案"""
        # 按句子分割
        sentences = re.split(r'(?<=[。！？；])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
        
        # 简单策略：取开头、中间和结尾的句子
        key_sentences = []
        key_sentences.append(sentences[0])  # 开头
        
        if num_sentences > 1:
            mid_idx = len(sentences) // 2
            key_sentences.append(sentences[mid_idx])  # 中间
        
        if num_sentences > 2:
            key_sentences.append(sentences[-1])  # 结尾
        
        return " ".join(key_sentences)
    
    def _process_long_answer(self, answer_text: str, llm_client) -> str:
        """
        处理超长答案：链式摘要 + 蕴含校验
        """
        if len(answer_text) <= self.chain_summary_threshold:
            return answer_text
        
        self.logger.info(f"Processing long answer ({len(answer_text)} chars) with chain summarization")
        
        # 将长答案分成多个片段
        chunk_size = self.chain_summary_threshold
        chunks = []
        
        for i in range(0, len(answer_text), chunk_size):
            chunk = answer_text[i:i + chunk_size]
            chunks.append(chunk)
        
        self.logger.debug(f"Split long answer into {len(chunks)} chunks")
        
        # 对每个片段生成摘要并进行蕴含校验
        processed_parts = []
        
        for i, chunk in enumerate(chunks):
            # 生成摘要
            summary = self._generate_summary_with_llm(chunk, llm_client)
            
            if summary and i < len(chunks) - 1:  # 不是最后一个片段
                # 进行蕴含校验
                next_chunk_preview = chunks[i + 1][:500]  # 下一个片段的预览
                is_entailed, score = self._check_entailment(
                    premise=next_chunk_preview,
                    hypothesis=summary
                )
                
                if not is_entailed:
                    self.logger.warning(f"Chunk {i} summary failed entailment check (score: {score:.3f})")
                    # 使用抽取式摘要作为备用
                    summary = self._extract_key_sentences(chunk, num_sentences=3)
                else:
                    self.logger.debug(f"Chunk {i} summary passed entailment check (score: {score:.3f})")
            
            # 如果是最后一个片段，保留更多原始内容
            if i == len(chunks) - 1:
                # 最后一个片段可能包含结论，保留更多内容
                if len(chunk) > 1000:
                    summary = self._generate_summary_with_llm(chunk[:1000], llm_client) + "\n\n" + chunk[-500:]
                else:
                    summary = chunk
            
            if summary:
                processed_parts.append(summary)
        
        # 合并处理后的部分
        final_answer = "\n\n".join(processed_parts)
        
        self.logger.info(f"Long answer processing completed: {len(answer_text)} -> {len(final_answer)} chars")
        
        return final_answer
    
    def process_groups(self, groups: List[Dict[str, Any]], llm_client) -> List[Dict[str, Any]]:
        """
        处理语义分组后的块，支持长答案处理
        
        Args:
            groups: 语义分组后的块列表
            llm_client: LLM客户端实例
            
        Returns:
            处理后的QA对列表
        """
        all_qa_pairs = []
        
        for group_idx, group in enumerate(groups):
            try:
                group_content = group['content']
                confidence = group.get('confidence', 'unknown')
                
                # 根据置信度决定处理策略
                if confidence == 'high':
                    # 高置信度块，直接提取
                    self.logger.debug(f"Processing high confidence group {group_idx}")
                    qa_pairs = self._extract_from_high_confidence_block(group_content)
                else:
                    # 中低置信度块，使用LLM提取
                    self.logger.debug(f"Processing {confidence} confidence group {group_idx}")
                    qa_pairs = self._extract_with_llm(group_content, llm_client)
                
                # 处理长答案
                for pair in qa_pairs:
                    if len(pair.get('answer', '')) > self.chain_summary_threshold:
                        original_answer = pair['answer']
                        processed_answer = self._process_long_answer(original_answer, llm_client)
                        pair['answer'] = processed_answer
                        pair['original_answer_length'] = len(original_answer)
                        pair['processed_answer_length'] = len(processed_answer)
                        self.logger.info(f"Processed long answer: {len(original_answer)} -> {len(processed_answer)} chars")
                
                # 添加元数据
                for pair in qa_pairs:
                    pair['source_confidence'] = confidence
                    pair['source_type'] = group.get('type', 'unknown')
                    pair['domain'] = group.get('domain', 'general')
                
                all_qa_pairs.extend(qa_pairs)
                
            except Exception as e:
                self.logger.error(f"Error processing group {group_idx}: {e}")
                continue
        
        return all_qa_pairs
    
    def _extract_from_high_confidence_block(self, content: str) -> List[Dict[str, Any]]:
        """从高置信度块中提取问答对（基于规则）"""
        qa_pairs = []
        
        # 简单的基于模式的提取
        lines = content.split('\n')
        current_question = None
        current_answer_lines = []
        
        question_patterns = [
            r"^网友[：:](.+)$",
            r"^记者[：:](.+)$",
            r"^问[：:](.+)$",
            r"^提问[：:](.+)$",
            r"^Q[：:](.+)$"
        ]
        
        answer_patterns = [
            r"^段永平[：:](.+)$",
            r"^段[：:](.+)$",
            r"^答[：:](.+)$",
            r"^A[：:](.+)$"
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是问题
            question_match = None
            for pattern in question_patterns:
                match = re.match(pattern, line)
                if match:
                    question_match = match.group(1).strip()
                    break
            
            if question_match:
                # 保存之前的问答对
                if current_question and current_answer_lines:
                    qa_pairs.append({
                        'question': current_question,
                        'answer': '\n'.join(current_answer_lines)
                    })
                
                current_question = question_match
                current_answer_lines = []
                continue
            
            # 检查是否是答案开始
            answer_match = None
            for pattern in answer_patterns:
                match = re.match(pattern, line)
                if match:
                    answer_match = match.group(1).strip()
                    break
            
            if answer_match and current_question:
                current_answer_lines.append(answer_match)
            elif current_question and current_answer_lines:
                # 继续收集答案内容
                current_answer_lines.append(line)
        
        # 保存最后的问答对
        if current_question and current_answer_lines:
            qa_pairs.append({
                'question': current_question,
                'answer': '\n'.join(current_answer_lines)
            })
        
        return qa_pairs
    
    def _extract_with_llm(self, content: str, llm_client) -> List[Dict[str, Any]]:
        """使用LLM提取问答对"""
        # 创建prompt
        prompt = self.create_prompt(content)
        
        # 调用LLM
        response = llm_client.call_ollama(prompt, temperature=0.1)
        
        if not response:
            return []
        
        # 提取JSON
        qa_pairs = self.extract_json(response)
        
        return qa_pairs 