"""Q&A extraction module for processing and extracting question-answer pairs."""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .prompt_generator import PromptGenerator

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
        
        # 初始化 prompt 生成器
        prompt_template_dir = self.config.get('prompt_template_dir', 'src/prompts')
        self.prompt_generator = PromptGenerator(prompt_template_dir)
        
        # 加载目标人物配置
        person_config_path = self.config.get('target_person_config')
        if person_config_path:
            self.person_config = self.prompt_generator.load_person_config(person_config_path)
        else:
            # 如果没有指定配置文件，使用默认配置
            self.person_config = self.prompt_generator.load_person_config('config/target_persons/duan_yongping.yaml')
        
        # 生成 prompts
        self.compact_prompt = self.prompt_generator.generate_compact_prompt(self.person_config)
        self.full_prompt = self.prompt_generator.generate_full_prompt(self.person_config)
        
        # 获取目标人物信息用于规则提取
        self.target_person = self.person_config['target_person']
        self.questioner_prefixes = self.person_config.get('questioner_types', [])
        
        # 构建回答者前缀列表（主名称 + 所有别名）
        self.answerer_prefixes = [self.target_person['main_name']] + self.target_person.get('aliases', [])
    
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
        sentences = re.split(r'(?<=[。！？；])', text)
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
        🔥 改进版本：支持更多置信度级别和灵活处理策略
        
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
                
                self.logger.debug(f"Processing group {group_idx + 1}/{len(groups)} - confidence: {confidence}")
                
                # 🔥 根据置信度决定处理策略
                if confidence == 'high':
                    # 高置信度块：直接使用规则提取
                    self.logger.debug(f"Processing high confidence group {group_idx} with rule-based extraction")
                    qa_pairs = self._extract_from_high_confidence_block(group_content)
                    
                elif confidence == 'medium':
                    # 🔥 中置信度块：先尝试规则提取，再用LLM补充
                    self.logger.debug(f"Processing medium confidence group {group_idx} with hybrid approach")
                    
                    # 先尝试规则提取
                    rule_pairs = self._extract_from_high_confidence_block(group_content)
                    
                    if rule_pairs:
                        # 如果规则提取成功，使用规则结果
                        qa_pairs = rule_pairs
                        self.logger.debug(f"Medium confidence block processed successfully with rules: {len(rule_pairs)} pairs")
                    else:
                        # 如果规则提取失败，使用LLM处理
                        self.logger.debug(f"Medium confidence block fallback to LLM processing")
                        qa_pairs = self._extract_with_llm_enhanced(group_content, llm_client, confidence)
                
                else:
                    # 低置信度块：使用改进的LLM处理
                    self.logger.debug(f"Processing {confidence} confidence group {group_idx} with enhanced LLM")
                    qa_pairs = self._extract_with_llm_enhanced(group_content, llm_client, confidence)
                
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
                    pair['group_index'] = group_idx
                
                all_qa_pairs.extend(qa_pairs)
                
                if qa_pairs:
                    self.logger.info(f"✅ Group {group_idx + 1}: Extracted {len(qa_pairs)} Q&A pairs (confidence: {confidence})")
                else:
                    self.logger.debug(f"❌ Group {group_idx + 1}: No Q&A pairs extracted (confidence: {confidence})")
                
            except Exception as e:
                self.logger.error(f"Error processing group {group_idx}: {e}")
                continue
        
        self.logger.info(f"Total QA pairs extracted: {len(all_qa_pairs)}")
        return all_qa_pairs
    
    def _extract_from_high_confidence_block(self, content: str) -> List[Dict[str, Any]]:
        """
        从高置信度块中提取问答对（基于规则）- 大幅改进版本
        
        主要改进：
        1. 🔥 更完整的问答对识别：确保不遗漏任何问答对
        2. 🔥 支持多种问答组合模式：问-答、问-问-答、问-答-答等
        3. 🔥 改进配对逻辑：更智能的问答配对策略
        4. 质量验证和内容清理
        5. 支持带标识符的说话人模式
        """
        
        # 使用配置的说话人前缀
        questioner_prefixes = self.questioner_prefixes
        answerer_prefixes = self.answerer_prefixes
        
        # 预处理：去除编号前缀（如"08. 网友："）
        content = re.sub(r'(?:^|\n)\d+\.\s*', '\n', content, flags=re.MULTILINE)
        
        # 构建更精确的正则表达式
        questioner_patterns = []
        answerer_patterns = []
        
        for prefix in questioner_prefixes:
            # 模式1: 前缀 + 可选空格 + 可选标识符 (网友O, 网友 A, 记者 123等)
            questioner_patterns.append(f'{re.escape(prefix)}\\s*[A-Za-z0-9\u4e00-\u9fa5]*')
            # 模式2: 标识符 + 空格 + 前缀 (A 网友, sam 观众等)
            questioner_patterns.append(f'[A-Za-z0-9\u4e00-\u9fa5]+\\s+{re.escape(prefix)}')
        
        for prefix in answerer_prefixes:
            answerer_patterns.append(f'{re.escape(prefix)}\\s*[A-Za-z0-9\u4e00-\u9fa5]*')
            answerer_patterns.append(f'[A-Za-z0-9\u4e00-\u9fa5]+\\s+{re.escape(prefix)}')
        
        all_patterns = questioner_patterns + answerer_patterns
        
        # 匹配：行首或换行后的说话人模式，后面跟冒号
        pattern = re.compile(f'(?:^|\\n)\\s*({"|".join(all_patterns)})\\s*[:：]', re.MULTILINE)
        
        matches = list(pattern.finditer(content))
        
        if not matches:
            return []
        
        # 🔥 改进：先收集所有的对话段
        segments = []
        for i, match in enumerate(matches):
            speaker = match.group(1).strip()
            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            content_text = content[content_start:content_end].strip()
            
            if content_text.strip():
                segments.append({
                    'speaker': speaker,
                    'content': content_text,
                    'is_questioner': self._is_questioner(speaker, questioner_prefixes),
                    'is_answerer': self._is_answerer(speaker, answerer_prefixes),
                    'position': i
                })
        
        # 🔥 改进：智能配对逻辑
        qa_pairs = []
        i = 0
        while i < len(segments):
            segment = segments[i]
            
            # 如果当前段是问题
            if segment['is_questioner']:
                current_qa = {
                    "question": segment['content'],
                    "answer": "",
                    "source_confidence": "high", 
                    "source_type": "rule_based_strict"
                }
                
                # 🔥 寻找对应的答案：在接下来的段落中寻找
                answer_found = False
                j = i + 1
                
                # 最多向前查找3个段落寻找答案
                while j < len(segments) and j < i + 4:
                    next_segment = segments[j]
                    
                    # 如果找到答案者的发言
                    if next_segment['is_answerer']:
                        current_qa["answer"] = next_segment['content']
                        answer_found = True
                        
                        # 🔥 检查是否有后续的补充答案（同一回答者的继续发言）
                        k = j + 1
                        while k < len(segments) and k < j + 3:
                            follow_up = segments[k]
                            # 如果下一段还是答案者，或者是无标识的文本（可能是答案的延续）
                            if (follow_up['is_answerer'] or 
                                (not follow_up['is_questioner'] and not follow_up['is_answerer'] and 
                                 len(follow_up['content']) > 20)):
                                current_qa["answer"] += "\n\n" + follow_up['content']
                                k += 1
                            else:
                                break
                        
                        break
                    
                    # 如果遇到新的问题，停止寻找答案
                    elif next_segment['is_questioner']:
                        break
                    
                    j += 1
                
                # 🔥 如果找到了有效的问答对，添加到结果中
                if answer_found and self._is_valid_qa_pair_rule_based(current_qa):
                    qa_pairs.append(self._finalize_qa_pair(current_qa))
                    self.logger.debug(f"Successfully extracted Q-A pair: {current_qa['question'][:50]}... -> {current_qa['answer'][:50]}...")
                else:
                    # 🔥 即使没找到答案，如果问题本身有价值，也尝试收集
                    if len(current_qa['question']) > 10:
                        # 作为单独的问题记录（可能后续会有LLM处理）
                        self.logger.debug(f"Found question without immediate answer: {current_qa['question'][:50]}...")
            
            # 🔥 如果当前段是答案，但前面没有匹配的问题（可能是独立回答）
            elif segment['is_answerer']:
                # 向前查找是否有未配对的问题
                unmatched_question = None
                for k in range(max(0, i-3), i):
                    if (segments[k]['is_questioner'] and 
                        not any(pair['question'].strip() == segments[k]['content'].strip() for pair in qa_pairs)):
                        unmatched_question = segments[k]
                        break
                
                if unmatched_question:
                    orphan_qa = {
                        "question": unmatched_question['content'],
                        "answer": segment['content'],
                        "source_confidence": "high",
                        "source_type": "rule_based_strict"
                    }
                    
                    if self._is_valid_qa_pair_rule_based(orphan_qa):
                        qa_pairs.append(self._finalize_qa_pair(orphan_qa))
                        self.logger.debug(f"Rescued orphan Q-A pair: {orphan_qa['question'][:50]}... -> {orphan_qa['answer'][:50]}...")
            
            i += 1
        
        self.logger.info(f"🔥 Enhanced extraction completed: {len(qa_pairs)} Q-A pairs found")
        return qa_pairs

    def _clean_content(self, content: str) -> str:
        """清理内容文本"""
        # 移除时间戳（支持中文和英文圆括号）
        content = re.sub(r'[\(（][0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}[\)）]', '', content)
        
        # 移除多余的空白和换行
        content = re.sub(r'\s+', ' ', content)
        
        # 移除开头的编号
        content = re.sub(r'^\d+\.\s*', '', content)
        
        return content.strip()

    def _is_valid_qa_pair_rule_based(self, qa: Dict[str, Any]) -> bool:
        """验证问答对是否有效"""
        if not qa or not qa.get('question') or not qa.get('answer'):
            return False
        
        # 对原始文本进行基本清理后再验证
        question = self._clean_content(qa['question'])
        answer = self._clean_content(qa['answer'])
        
        # 基本长度检查
        if len(question) < 5 or len(answer) < 5:
            return False
        
        # 检查问题中是否包含回答者前缀（带冒号的格式，避免误判）
        # 使用配置的回答者前缀
        answerer_pattern = r'(' + '|'.join(re.escape(prefix) for prefix in self.answerer_prefixes) + r')\s*[:：]'
        if re.search(answerer_pattern, question):
            return False
        
        # 检查答案中是否包含提问者前缀（带冒号的格式）
        # 使用配置的提问者前缀
        questioner_pattern = r'(' + '|'.join(re.escape(prefix) for prefix in self.questioner_prefixes) + r')\s*[:：]'
        if re.search(questioner_pattern, answer):
            return False
        
        return True

    def _finalize_qa_pair(self, qa: Dict[str, Any]) -> Dict[str, Any]:
        """最终处理问答对"""
        # 1. 先从原始文本中提取时间戳（必须在清理之前）
        timestamp_pattern = r'[\(（]([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2})[\)）]'
        timestamp_match = re.search(timestamp_pattern, qa.get('answer', ''))
        if timestamp_match:
            qa['timestamp'] = timestamp_match.group(1)
        else:
            # 也检查问题中是否有时间戳
            timestamp_match = re.search(timestamp_pattern, qa.get('question', ''))
            if timestamp_match:
                qa['timestamp'] = timestamp_match.group(1)
        
        # 2. 然后再清理文本（移除时间戳等）
        qa['question'] = self._clean_content(qa['question'])
        qa['answer'] = self._clean_content(qa['answer'])
        
        # 3. 添加话题（简单实现）
        text = qa['question'] + ' ' + qa['answer']
        topics = []
        investment_keywords = ['投资', '价值投资', '股票', '企业', '现金流', '护城河', '管理', '长期', '巴菲特', '估值', '财务', '分红', '商业模式', '竞争优势', '成长', '收益', '风险', '市场']
        for keyword in investment_keywords:
            if keyword in text:
                topics.append(keyword)
        
        qa['topic'] = ','.join(topics[:3]) if topics else 'general'
        qa['domain'] = 'general'
        
        return qa
    
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

    def _is_questioner(self, speaker: str, questioner_prefixes: List[str]) -> bool:
        """判断说话人是否为提问者"""
        speaker_lower = speaker.lower()
        for prefix in questioner_prefixes:
            if (speaker_lower.startswith(prefix.lower()) or 
                speaker_lower.endswith(prefix.lower()) or
                f' {prefix.lower()}' in speaker_lower or
                f'{prefix.lower()} ' in speaker_lower):
                return True
        return False
    
    def _is_answerer(self, speaker: str, answerer_prefixes: List[str]) -> bool:
        """判断说话人是否为回答者"""
        speaker_lower = speaker.lower()
        for prefix in answerer_prefixes:
            if (speaker_lower.startswith(prefix.lower()) or 
                speaker_lower.endswith(prefix.lower()) or
                f' {prefix.lower()}' in speaker_lower or
                f'{prefix.lower()} ' in speaker_lower):
                return True
        return False 

    def _extract_with_llm_enhanced(self, content: str, llm_client, confidence: str) -> List[Dict[str, Any]]:
        """
        🔥 增强的LLM提取方法，针对不同置信度优化
        """
        # 根据置信度调整处理策略
        if confidence == 'low':
            # 对于低置信度块，使用更宽松的prompt和参数
            # 构建目标人物的所有名称
            all_names = '/'.join(self.answerer_prefixes)
            enhanced_prompt = f"""你是专业问答对提取专家。以下文本可能包含问答对，请仔细分析并提取所有可能的问答。

【宽松提取策略】
• 问题可能没有明确标识符，根据语义判断
• 回答者可能是{all_names}或其他相关人员
• 允许间接问答形式
• 尽可能提取有价值的对话内容

输出格式：JSON数组 [{{"question": "问题", "answer": "回答"}}]

文本内容：
{content}"""
            
            response = llm_client.call_ollama(enhanced_prompt, temperature=0.2)  # 稍高温度增加灵活性
        else:
            # 使用标准prompt
            prompt = self.create_prompt(content)
            response = llm_client.call_ollama(prompt, temperature=0.1)
        
        if not response:
            return []
        
        # 提取JSON
        qa_pairs = self.extract_json(response)
        
        # 🔥 增强的后处理：对提取结果进行质量检查
        filtered_pairs = []
        for pair in qa_pairs:
            if self._validate_extracted_pair(pair, confidence):
                filtered_pairs.append(pair)
        
        return filtered_pairs
    
    def _validate_extracted_pair(self, pair: Dict[str, Any], confidence: str) -> bool:
        """
        🔥 验证提取的问答对质量
        根据置信度采用不同的验证标准
        """
        if not pair or not pair.get('question') or not pair.get('answer'):
            return False
        
        question = str(pair['question']).strip()
        answer = str(pair['answer']).strip()
        
        # 基本长度检查
        if len(question) < 3 or len(answer) < 3:
            return False
        
        # 🔥 宽松验证：对于低置信度块，降低验证标准
        if confidence == 'low':
            # 低置信度块：只要有基本的问答结构即可
            return len(question) >= 3 and len(answer) >= 10
        
        # 标准验证
        # 检查问题和答案不能完全相同
        if question.lower() == answer.lower():
            return False
        
        # 检查答案不能包含明显的问题标识符
        question_indicators = ['网友：', '问：', '记者：', '主持人：']
        for indicator in question_indicators:
            if indicator in answer:
                return False
        
        return True 