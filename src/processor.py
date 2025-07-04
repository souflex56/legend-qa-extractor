"""Main processor class that orchestrates the Q&A extraction workflow."""

import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .config import Config
from .core import PDFProcessor, TextProcessor, QAExtractor, LLMClient
from .core.smart_block_processor import SmartBlockProcessor
from .utils import setup_logger, setup_extraction_loggers, save_single_jsonl_item, ensure_dir


class QAExtractionProcessor:
    """Main processor class for Q&A extraction from PDF documents."""
    
    def __init__(self, config: Config):
        """Initialize the processor with configuration.
        
        Args:
            config: Configuration object containing all settings
        """
        self.config = config
        
        # Set up logging
        self.logger = setup_logger(
            "qa_extractor", 
            log_level=config.log_level,
            log_file=os.path.join(config.output_dir, "main.log")
        )
        
        # Set up specialized loggers
        if config.enable_error_log or config.enable_success_log:
            self.error_logger, self.success_logger = setup_extraction_loggers(config.output_dir)
        else:
            self.error_logger = self.success_logger = None
        
        # Initialize processors
        self.pdf_processor = PDFProcessor()
        self.text_processor = TextProcessor(known_prefixes=config.known_prefixes)
        self.qa_extractor = QAExtractor(max_prompt_tokens=config.max_prompt_tokens)
        
        # Initialize LLM client
        try:
            self.llm_client = LLMClient(
                host=config.ollama_host,
                model_name=config.model_name
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise
        
        # Initialize token monitoring
        self.token_stats = {
            'prompt_uses': {'compact': 0, 'full': 0},
            'token_usage': [],
            'truncations': 0,
            'max_token_usage': 0,
            'min_token_usage': float('inf'),
            'total_blocks_processed': 0
        }
        
        # Initialize SmartBlockProcessor
        self.smart_block_processor = SmartBlockProcessor(
            text_processor=self.text_processor,
            llm_client=self.llm_client,
            max_block_size=config.max_block_size,
            min_block_size=config.min_block_size,
            qa_allowance_ratio=config.qa_allowance_ratio,
            enable_sliding_context=config.enable_sliding_context,
            enable_llm_anchor=config.enable_llm_anchor,
            anchor_keywords_count=config.anchor_keywords_count
        )
        
        self.logger.info("QA Extraction Processor initialized successfully")
    
    def process_pdf(self, pdf_path: Optional[str] = None) -> Dict[str, Any]:
        """Process a PDF file and extract Q&A pairs.
        
        Args:
            pdf_path: Path to PDF file. If None, uses config.pdf_filename
            
        Returns:
            Dictionary containing processing results and statistics
        """
        # Determine PDF path
        if pdf_path is None:
            pdf_path = self.config.pdf_filename
        
        if not os.path.isabs(pdf_path):
            # If relative path, look in current directory
            pdf_path = os.path.abspath(pdf_path)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"🔎 Starting processing of file: {pdf_path}")
        
        # Extract text from PDF
        self.logger.info("📄 Extracting text from PDF...")
        raw_text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        self.logger.info(f"📄 Extracted {len(raw_text)} characters from PDF")
        
        # Get PDF info
        pdf_info = self.pdf_processor.get_pdf_info(pdf_path)
        self.logger.info(f"📊 PDF info: {pdf_info.get('page_count', 'unknown')} pages")
        
        # Process text and create blocks
        self.logger.info("✂️ Starting text block generation using SmartBlockProcessor...")
        
        # 预处理文本
        preprocessed_text = self.text_processor.preprocess_qa_text(raw_text)
        
        # **重构改动**: 调用新的智能分块器，但暂时不生成LLM anchor
        # 临时禁用LLM anchor功能，稍后为问答对生成主题
        original_enable_llm_anchor = self.smart_block_processor.enable_llm_anchor
        self.smart_block_processor.enable_llm_anchor = False
        
        processed_blocks_data = self.smart_block_processor.process_document_into_blocks(preprocessed_text)
        
        # 恢复原始设置
        self.smart_block_processor.enable_llm_anchor = original_enable_llm_anchor
        
        self.logger.info(f"✅ Generated {len(processed_blocks_data)} smart text blocks for LLM processing.")
        
        # --- BEGIN: Added for block size inspection ---
        self.logger.info("🔍 Individual Block Sizes:")
        total_chars = 0
        for i, block_data in enumerate(processed_blocks_data):
            size = len(block_data.get('content', ''))
            total_chars += size
            self.logger.info(f"  - Block {i+1}/{len(processed_blocks_data)}: {size} characters")
        if processed_blocks_data:
            avg_size = total_chars / len(processed_blocks_data)
            self.logger.info(f"  - Average block size: {avg_size:.0f} characters")
        # --- END: Added for block size inspection ---
        
        # Filter blocks if QA filtering is enabled - 现在操作的是包含元数据的块字典列表
        if self.config.enable_qa_filter:
            original_count = len(processed_blocks_data)
            processed_blocks_data = [b for b in processed_blocks_data if self.text_processor.block_has_qa(b["content"])]
            self.logger.info(f"⚡ QA filtering: {len(processed_blocks_data)} blocks remaining (from {original_count})")
        
        # Apply sampling ratio - 现在操作的是包含元数据的块字典列表
        if self.config.extract_ratio < 1.0:
            sample_size = max(int(len(processed_blocks_data) * self.config.extract_ratio), 1)
            processed_blocks_data = processed_blocks_data[:sample_size]
            self.logger.info(f"⚡ Applied sampling ratio: {len(processed_blocks_data)} blocks selected")
        
        if not processed_blocks_data:
            self.logger.warning("⚠️ No valid blocks found for processing")
            return {
                'success': False,
                'message': 'No valid blocks found for processing',
                'stats': {'total_blocks': 0, 'qa_pairs_extracted': 0}
            }
        
        # Prepare output
        output_path = self._get_output_path()
        ensure_dir(os.path.dirname(output_path))
        
        # Clear output file
        with open(output_path, "w", encoding="utf-8") as f:
            pass
        
        # Process blocks and extract Q&A pairs
        self.logger.info(f"🤖 Processing {len(processed_blocks_data)} blocks with LLM...")
        results = self._process_blocks(processed_blocks_data, output_path, original_enable_llm_anchor)
        
        # Generate final statistics
        stats = self._generate_statistics(results, pdf_info, len(processed_blocks_data))
        
        self.logger.info(f"🎉 Processing completed! Extracted {stats['qa_pairs_extracted']} Q&A pairs")
        self.logger.info(f"📁 Output saved to: {output_path}")
        
        # 输出token监控总结（如果启用）
        if self.config.enable_token_monitoring:
            self._log_token_monitoring_summary()
        
        return {
            'success': True,
            'output_path': output_path,
            'stats': stats,
            'pdf_info': pdf_info
        }
    
    def _process_blocks(self, blocks: List[Dict[str, Any]], output_path: str, enable_llm_anchor: bool) -> List[Dict[str, Any]]:
        """Process text blocks and extract Q&A pairs.
        
        Args:
            blocks: List of text blocks to process
            output_path: Path to save extracted Q&A pairs
            
        Returns:
            List of processing results for each block
        """
        results = []
        success_count = 0
        
        for block_idx, block_data in enumerate(tqdm(blocks, desc="Extracting Q&A pairs")):
            # **关键改动**: 提取块内容和元数据
            block_content = block_data["content"]
            block_anchor = block_data.get("anchor", "")
            sliding_context = block_data.get("sliding_context", "")
            
            # Preprocess text
            processed_block = self.text_processor.preprocess_qa_text(block_content)
            
            # **关键改动**: 使用新的 prompt 格式，传入上下文和锚点
            prompt = self.qa_extractor.create_prompt(
                processed_block,
                sliding_context=sliding_context,
                block_anchor=block_anchor
            )
            
            # Token监控：记录使用情况（如果启用）
            if self.config.enable_token_monitoring:
                self._track_token_usage(prompt, block_anchor, sliding_context)
            
            # Call LLM
            response = self.llm_client.call_ollama(
                prompt, 
                temperature=self.config.temperature
            )
            
            if response is None:
                self.logger.warning(f"❌ Block {block_idx + 1}: LLM call failed")
                if self.error_logger:
                    self.error_logger.error(
                        f"LLM call failed for block {block_idx + 1}\n"
                        f"Block content:\n{block_content}"
                    )
                results.append({
                    'block_idx': block_idx,
                    'success': False,
                    'error': 'LLM call failed',
                    'qa_count': 0
                })
                continue
            
            # Extract Q&A pairs
            qa_pairs = self.qa_extractor.extract_json(response)
            
            if not qa_pairs:
                self.logger.warning(f"❌ Block {block_idx + 1}: No Q&A pairs extracted")
                if self.error_logger:
                    self.error_logger.error(
                        f"No valid Q&A pairs extracted for block {block_idx + 1}\n"
                        f"LLM response: {response}\n"
                        f"Block content:\n{block_content}"
                    )
                results.append({
                    'block_idx': block_idx,
                    'success': False,
                    'error': 'No Q&A pairs extracted',
                    'qa_count': 0
                })
                continue
            
            # Process and save Q&A pairs
            processed_pairs = self.qa_extractor.process_qa_pairs(
                qa_pairs, block_content, self.text_processor
            )
            
            # **重构改动**: 为每个Q&A对生成主题关键词（而不是使用文本块的anchor）
            for pair in processed_pairs:
                # 添加滑动上下文（如果启用）
                if sliding_context:
                    pair["sliding_context"] = sliding_context
                
                # **关键改动**: 为每个问答对生成主题，而不是使用文本块的anchor
                if enable_llm_anchor:
                    qa_topic = self._generate_qa_topic(pair["question"], pair["answer"])
                    if qa_topic:
                        pair["topic"] = qa_topic
                        self.logger.debug(f"Generated topic for Q&A: {qa_topic}")
            
            # Save each Q&A pair
            for pair in processed_pairs:
                save_single_jsonl_item(pair, output_path)
            
            # Log success
            self.logger.info(f"✅ Block {block_idx + 1}: Extracted {len(processed_pairs)} Q&A pairs")
            if self.success_logger:
                for i, pair in enumerate(processed_pairs):
                    success_log_content = (
                        f"Successfully extracted Q&A pair #{success_count + i + 1} from block {block_idx + 1}:\n\n"
                        f"Question: {pair['question']}\n\n"
                        f"Answer: {pair['answer']}\n\n"
                        f"Source block:\n{block_content}\n\n"
                        f"{'='*80}"
                    )
                    self.success_logger.info(success_log_content)
            
            success_count += len(processed_pairs)
            
            results.append({
                'block_idx': block_idx,
                'success': True,
                'qa_count': len(processed_pairs),
                'qa_pairs': processed_pairs
            })
        
        return results
    
    def _get_output_path(self) -> str:
        """Get the full output file path."""
        if os.path.isabs(self.config.output_filename):
            return self.config.output_filename
        
        return os.path.join(self.config.output_dir, self.config.output_filename)
    
    def _generate_statistics(self, results: List[Dict[str, Any]], 
                           pdf_info: Dict[str, Any],
                           total_blocks: int) -> Dict[str, Any]:
        """Generate processing statistics.
        
        Args:
            results: List of processing results
            pdf_info: PDF information
            total_blocks: Total number of blocks processed
            
        Returns:
            Statistics dictionary
        """
        successful_blocks = sum(1 for r in results if r['success'])
        total_qa_pairs = sum(r['qa_count'] for r in results)
        
        # Calculate quality metrics if we have successful extractions
        quality_metrics = {}
        if successful_blocks > 0:
            all_qa_pairs = []
            for r in results:
                if r['success'] and 'qa_pairs' in r:
                    all_qa_pairs.extend(r['qa_pairs'])
            
            if all_qa_pairs:
                question_lengths = [len(pair['question']) for pair in all_qa_pairs]
                answer_lengths = [len(pair['answer']) for pair in all_qa_pairs]
                
                quality_metrics = {
                    'avg_question_length': sum(question_lengths) / len(question_lengths),
                    'avg_answer_length': sum(answer_lengths) / len(answer_lengths),
                    'min_question_length': min(question_lengths),
                    'max_question_length': max(question_lengths),
                    'min_answer_length': min(answer_lengths),
                    'max_answer_length': max(answer_lengths)
                }
        
        return {
            'total_blocks': total_blocks,
            'successful_blocks': successful_blocks,
            'failed_blocks': total_blocks - successful_blocks,
            'success_rate': successful_blocks / total_blocks if total_blocks > 0 else 0,
            'qa_pairs_extracted': total_qa_pairs,
            'avg_qa_per_block': total_qa_pairs / successful_blocks if successful_blocks > 0 else 0,
            'pdf_pages': pdf_info.get('page_count', 0),
            'quality_metrics': quality_metrics,
            'config_used': {
                'model_name': self.config.model_name,
                'max_block_size': self.config.max_block_size,
                'min_block_size': self.config.min_block_size,
                'extract_ratio': self.config.extract_ratio,
                'enable_qa_filter': self.config.enable_qa_filter,
                'temperature': self.config.temperature
            }
        }
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate the setup and configuration.
        
        Returns:
            Validation results dictionary
        """
        validation = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check PDF file
        pdf_path = self.config.pdf_filename
        if not os.path.isabs(pdf_path):
            pdf_path = os.path.abspath(pdf_path)
        
        if not os.path.exists(pdf_path):
            validation['valid'] = False
            validation['issues'].append(f"PDF file not found: {pdf_path}")
        
        # Check LLM connection
        if not self.llm_client._test_connection():
            validation['valid'] = False
            validation['issues'].append("Cannot connect to Ollama server")
        
        # Check model availability
        if not self.llm_client.check_model_availability():
            validation['warnings'].append(f"Model {self.config.model_name} not found locally")
        
        # Check output directory permissions
        try:
            ensure_dir(self.config.output_dir)
        except Exception as e:
            validation['valid'] = False
            validation['issues'].append(f"Cannot create output directory: {e}")
        
        return validation
    
    def _generate_qa_topic(self, question: str, answer: str) -> str:
        """为问答对生成主题关键词"""
        try:
            # 组合问答对内容
            qa_content = f"问题: {question}\n答案: {answer}"
            
            # 调用SmartBlockProcessor的LLM anchor生成方法，但用于问答对
            anchor = self.smart_block_processor._generate_anchor_with_llm(qa_content)
            
            if anchor:
                self.logger.debug(f"Generated topic for Q&A pair: {anchor}")
                return anchor
            else:
                self.logger.warning("Failed to generate topic for Q&A pair")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error generating topic for Q&A pair: {e}")
            return ""
    
    def _log_token_monitoring_summary(self):
        """在处理完成后输出token使用总结"""
        if self.token_stats['total_blocks_processed'] == 0:
            return
        
        stats = self.token_stats
        avg_usage = sum(stats['token_usage']) / len(stats['token_usage']) if stats['token_usage'] else 0
        
        self.logger.info("📊 Token使用总结报告")
        self.logger.info("=" * 50)
        self.logger.info(f"🔢 处理块数: {stats['total_blocks_processed']}")
        self.logger.info(f"📝 Prompt使用统计:")
        self.logger.info(f"   精简版: {stats['prompt_uses']['compact']} 次")
        self.logger.info(f"   完整版: {stats['prompt_uses']['full']} 次")
        
        if stats['token_usage']:
            self.logger.info(f"🎯 Token使用统计:")
            self.logger.info(f"   平均使用: {avg_usage:.0f} tokens")
            self.logger.info(f"   最大使用: {stats['max_token_usage']} tokens")
            self.logger.info(f"   最小使用: {stats['min_token_usage']} tokens")
            
            utilization = avg_usage / self.config.max_prompt_tokens * 100
            self.logger.info(f"   平均利用率: {utilization:.1f}%")
            
            if utilization > 90:
                self.logger.warning("⚠️ Token利用率过高，建议优化配置")
            elif utilization > 75:
                self.logger.info("🟡 Token利用率较高，建议监控")
            else:
                self.logger.info("🟢 Token利用率健康")
        
        if stats['truncations'] > 0:
            self.logger.warning(f"⚠️ 发生 {stats['truncations']} 次文本截断")
        else:
            self.logger.info("✅ 无文本截断发生")
        
        self.logger.info("=" * 50)
    
    def _track_token_usage(self, prompt: str, block_anchor: str, sliding_context: str):
        """记录token使用情况用于后续分析"""
        try:
            # 估算token使用
            token_count = self.qa_extractor.estimate_token_count(prompt)
            
            # 更新统计
            self.token_stats['token_usage'].append(token_count)
            self.token_stats['max_token_usage'] = max(self.token_stats['max_token_usage'], token_count)
            self.token_stats['min_token_usage'] = min(self.token_stats['min_token_usage'], token_count)
            self.token_stats['total_blocks_processed'] += 1
            
            # 判断使用的prompt类型
            if self.qa_extractor.compact_prompt in prompt:
                self.token_stats['prompt_uses']['compact'] += 1
            else:
                self.token_stats['prompt_uses']['full'] += 1
            
            # 检查是否可能发生截断
            if token_count > self.config.max_prompt_tokens:
                self.token_stats['truncations'] += 1
                self.logger.warning(f"⚠️ Potential truncation detected: {token_count} tokens > {self.config.max_prompt_tokens} limit")
            
            # 详细日志记录
            self.logger.debug(f"Block token usage: {token_count}/{self.config.max_prompt_tokens} tokens ({token_count/self.config.max_prompt_tokens*100:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Error tracking token usage: {e}")