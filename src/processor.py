"""Main processor class that orchestrates the Q&A extraction workflow."""

import os
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time

from .config import Config
from .core import PDFProcessor, TextProcessor, QAExtractor, LLMClient
from .core.semantic_grouper import SemanticGrouper
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
        self.qa_extractor = QAExtractor(max_prompt_tokens=config.max_prompt_tokens, config=config.to_dict())
        
        # Initialize LLM client
        try:
            self.llm_client = LLMClient(
                host=config.ollama_host,
                model_name=config.model_name
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise
        
        # Initialize Semantic Grouper
        self.semantic_grouper = SemanticGrouper(config.to_dict())
        

        
        # **🚀 PERFORMANCE: Batch processing configuration**
        self.batch_size = getattr(config, 'batch_size', 5)  # Default batch size
        self.max_workers = getattr(config, 'max_workers', 3)  # Conservative default
        
        # Token monitoring (如果启用)
        if config.enable_token_monitoring:
            self.token_usage_stats = []
        
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
        
        # **🚀 PERFORMANCE OPTIMIZATION: Model Warmup**
        if not self.llm_client.is_warmed_up():
            self.logger.info("🔥 Performing model warmup to eliminate cold start delays...")
            warmup_success = self.llm_client.warmup_model()
            if not warmup_success:
                self.logger.warning("⚠️ Model warmup failed, but continuing with processing...")
        else:
            self.logger.info("✅ Model already warmed up, proceeding with processing...")
        
        # Extract text from PDF
        self.logger.info("📄 Extracting text from PDF...")
        raw_text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        self.logger.info(f"📄 Extracted {len(raw_text)} characters from PDF")
        
        # Get PDF info
        pdf_info = self.pdf_processor.get_pdf_info(pdf_path)
        self.logger.info(f"📊 PDF info: {pdf_info.get('page_count', 'unknown')} pages")
        
        # Process text and create blocks
        self.logger.info("✂️ Starting semantic grouping...")
        
        # 预处理文本
        preprocessed_text = self.text_processor.preprocess_qa_text(raw_text)
        
        # 分割成段落
        paragraphs = [p.strip() for p in preprocessed_text.split('\n') if p.strip()]
        
        # 使用新的语义分组器
        processed_blocks_data = self.semantic_grouper.group(paragraphs)
        
        self.logger.info(f"✅ Generated {len(processed_blocks_data)} semantic blocks for LLM processing.")
        
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
            
            # 根据采样策略选择blocks
            if getattr(self.config, 'sampling_strategy', 'sequential') == 'random':
                import random
                # 随机采样
                processed_blocks_data = random.sample(processed_blocks_data, sample_size)
                self.logger.info(f"⚡ Applied random sampling ratio: {len(processed_blocks_data)} blocks selected")
            else:
                # 顺序采样（从头开始）
                processed_blocks_data = processed_blocks_data[:sample_size]
                self.logger.info(f"⚡ Applied sequential sampling ratio: {len(processed_blocks_data)} blocks selected")
        
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
        results = self._process_blocks(processed_blocks_data, output_path, self.config.enable_llm_anchor)
        
        # Generate final statistics with confidence analysis
        stats = self._generate_statistics(results, pdf_info, len(processed_blocks_data))
        
        # Add confidence-based processing statistics
        confidence_stats = self._analyze_confidence_processing(results, processed_blocks_data)
        stats.update(confidence_stats)
        

        
        self.logger.info(f"🎉 Processing completed! Extracted {stats['qa_pairs_extracted']} Q&A pairs")
        self.logger.info(f"📊 Confidence-based processing stats:")
        self.logger.info(f"   - High confidence blocks: {confidence_stats.get('high_confidence_blocks', 0)} (rule-based)")
        self.logger.info(f"   - Medium confidence blocks: {confidence_stats.get('medium_confidence_blocks', 0)} (LLM conservative)")
        self.logger.info(f"   - Low confidence blocks: {confidence_stats.get('low_confidence_blocks', 0)} (LLM permissive)")
        self.logger.info(f"   - Skipped blocks: {confidence_stats.get('skipped_blocks', 0)}")
        self.logger.info(f"📁 Output saved to: {output_path}")
        
        # 输出token监控总结（如果启用）
        if self.config.enable_token_monitoring:
            self._log_token_monitoring_summary()
        
        # **🚀 PERFORMANCE OPTIMIZATION: Log performance summary**
        self.llm_client.log_performance_summary()
        
        return {
            'success': True,
            'output_path': output_path,
            'stats': stats,
            'pdf_info': pdf_info,
            'performance_stats': self.llm_client.get_performance_report()
        }
    
    def _process_blocks(self, blocks: List[Dict[str, Any]], output_path: str, enable_llm_anchor: bool) -> List[Dict[str, Any]]:
        """Process text blocks and extract Q&A pairs using batch parallel processing.
        
        Args:
            blocks: List of text blocks to process
            output_path: Path to save extracted Q&A pairs
            enable_llm_anchor: Whether to generate LLM anchors for Q&A pairs
            
        Returns:
            List of processing results for each block
        """
        # **🚀 PERFORMANCE OPTIMIZATION: Initialize token monitoring for batch**
        if self.config.enable_token_monitoring:
            self.token_usage_stats.append({
                'prompt_uses': {'compact': 0, 'full': 0},
                'token_usage': [],
                'truncations': 0,
                'max_token_usage': 0,
                'min_token_usage': float('inf'),
                'total_blocks_processed': 0
            })
        
        # **🔥 CRITICAL FIX: Maintain model warmth to prevent cold starts during long processing**
        self.logger.info("🔥 Ensuring model warmth before processing...")
        warmth_maintained = self.llm_client.maintain_model_warmth()
        if not warmth_maintained:
            self.logger.warning("⚠️ Model warmth maintenance failed, may experience cold starts")
        
        # **🚀 PERFORMANCE: Decide between batch and serial processing**
        if len(blocks) >= self.batch_size and self.max_workers > 1:
            self.logger.info(f"🚀 Using batch parallel processing (batch_size={self.batch_size}, workers={self.max_workers})")
            return self._process_blocks_parallel(blocks, output_path, enable_llm_anchor)
        else:
            self.logger.info("📝 Using serial processing (small dataset or single worker)")
            return self._process_blocks_serial(blocks, output_path, enable_llm_anchor)
    
    def _process_blocks_parallel(self, blocks: List[Dict[str, Any]], output_path: str, enable_llm_anchor: bool) -> List[Dict[str, Any]]:
        """Process blocks in parallel batches for maximum performance.
        
        Args:
            blocks: List of text blocks to process
            output_path: Path to save extracted Q&A pairs
            enable_llm_anchor: Whether to generate LLM anchors for Q&A pairs
            
        Returns:
            List of processing results for each block
        """
        all_results = []
        
        # Split blocks into batches
        batches = [blocks[i:i + self.batch_size] for i in range(0, len(blocks), self.batch_size)]
        
        # Process batches with progress bar
        with tqdm(total=len(blocks), desc="🚀 Batch Processing Q&A Extraction") as pbar:
            for batch_idx, batch in enumerate(batches):
                self.logger.debug(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} blocks")
                
                # **🔥 CRITICAL FIX: Send keep-alive ping every 5 batches to prevent cold starts**
                if batch_idx > 0 and batch_idx % 5 == 0:
                    self.logger.debug(f"🔥 Sending keep-alive ping after {batch_idx} batches...")
                    self.llm_client.send_keepalive_ping()
                
                # Process batch in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all tasks in the batch
                    future_to_block = {
                        executor.submit(self._process_single_block, block_data, block_idx + batch_idx * self.batch_size, enable_llm_anchor): 
                        (block_data, block_idx + batch_idx * self.batch_size)
                        for block_idx, block_data in enumerate(batch)
                    }
                    
                    # Collect results as they complete
                    batch_results = []
                    for future in concurrent.futures.as_completed(future_to_block):
                        block_data, original_idx = future_to_block[future]
                        try:
                            result = future.result()
                            result['original_idx'] = original_idx  # Keep track of original order
                            batch_results.append(result)
                            
                            # Save Q&A pairs immediately if successful
                            if result['success'] and 'qa_pairs' in result:
                                for pair in result['qa_pairs']:
                                    save_single_jsonl_item(pair, output_path)
                                
                                # Log success
                                self.logger.info(f"✅ Block {original_idx + 1}: Extracted {result['qa_count']} Q&A pairs")
                                
                        except Exception as e:
                            self.logger.error(f"❌ Block {original_idx + 1}: Processing failed with exception: {e}")
                            batch_results.append({
                                'block_idx': original_idx,
                                'original_idx': original_idx,
                                'success': False,
                                'error': f'Processing exception: {e}',
                                'qa_count': 0
                            })
                        
                        pbar.update(1)
                
                # Sort results by original index to maintain order
                batch_results.sort(key=lambda x: x['original_idx'])
                all_results.extend(batch_results)
        
        return all_results
    
    def _process_blocks_serial(self, blocks: List[Dict[str, Any]], output_path: str, enable_llm_anchor: bool) -> List[Dict[str, Any]]:
        """Process blocks serially (fallback method).
        
        Args:
            blocks: List of text blocks to process
            output_path: Path to save extracted Q&A pairs
            enable_llm_anchor: Whether to generate LLM anchors for Q&A pairs
            
        Returns:
            List of processing results for each block
        """
        results = []
        
        for block_idx, block_data in enumerate(tqdm(blocks, desc="📝 Serial Processing Q&A Extraction")):
            result = self._process_single_block(block_data, block_idx, enable_llm_anchor)
            results.append(result)
            
            # Save Q&A pairs immediately if successful
            if result['success'] and 'qa_pairs' in result:
                for pair in result['qa_pairs']:
                    save_single_jsonl_item(pair, output_path)
                
                # Log success
                self.logger.info(f"✅ Block {block_idx + 1}: Extracted {result['qa_count']} Q&A pairs")
        
        return results
    
    def _process_single_block(self, block_data: Dict[str, Any], block_idx: int, enable_llm_anchor: bool) -> Dict[str, Any]:
        """Process a single text block and extract Q&A pairs.
        
        Args:
            block_data: Text block data with content and metadata
            block_idx: Index of the block being processed
            enable_llm_anchor: Whether to generate LLM anchors for Q&A pairs
            
        Returns:
            Processing result for the block
        """
        start_time = time.time()
        
        try:
            # Extract block content and metadata
            block_content = block_data["content"]
            # 新的语义分组器提供的元数据
            confidence = block_data.get("confidence", "unknown")
            block_type = block_data.get("type", "unknown")
            domain = block_data.get("domain", "general")
            
            # 暂时不使用sliding_context，因为新架构中的语义关联更强
            sliding_context = ""
            
            # Preprocess text
            processed_block = self.text_processor.preprocess_qa_text(block_content)
            
            # Create prompt with context information
            # 新架构中使用置信度和领域信息来优化prompt
            context_info = f"[置信度: {confidence}, 类型: {block_type}, 领域: {domain}]"
            prompt = self.qa_extractor.create_prompt(
                processed_block,
                sliding_context=sliding_context,
                block_anchor=context_info
            )
            
            # Token monitoring
            if self.config.enable_token_monitoring:
                self._track_token_usage(prompt, context_info, sliding_context)
            
            # 根据置信度决定处理策略
            if confidence == 'high':
                # 高置信度块：直接使用规则提取，无需LLM
                self.logger.debug(f"Processing high confidence block {block_idx + 1} with rule-based extraction")
                
                qa_pairs = self.qa_extractor._extract_from_high_confidence_block(processed_block)
            elif confidence == 'medium':
                # 中置信度块：使用完整LLM处理，但温度稍低
                self.logger.debug(f"Processing medium confidence block {block_idx + 1} with LLM (conservative)")
                
                response = self.llm_client.call_ollama(
                    prompt, 
                    temperature=max(0.05, self.config.temperature - 0.05)  # 稍微降低温度
                )
                
                if response is None:
                    self.logger.warning(f"❌ Block {block_idx + 1}: LLM call failed")
                    if self.error_logger:
                        self.error_logger.error(
                            f"LLM call failed for block {block_idx + 1}\n"
                            f"Block content:\n{block_content}"
                        )
                    return {
                        'block_idx': block_idx,
                        'success': False,
                        'error': 'LLM call failed',
                        'qa_count': 0
                    }
                
                qa_pairs = self.qa_extractor.extract_json(response)
                
                if not qa_pairs:
                    self.logger.warning(f"❌ Block {block_idx + 1}: No Q&A pairs extracted")
                    if self.error_logger:
                        self.error_logger.error(
                            f"No valid Q&A pairs extracted for block {block_idx + 1}\n"
                            f"LLM response: {response}\n"
                            f"Block content:\n{block_content}"
                        )
                    return {
                        'block_idx': block_idx,
                        'success': False,
                        'error': 'No Q&A pairs extracted',
                        'qa_count': 0
                    }
            else:
                # 低置信度块：使用更宽松的LLM处理，或跳过
                if confidence == 'low':
                    self.logger.debug(f"Processing low confidence block {block_idx + 1} with LLM (permissive)")
                    # 对于低置信度块，可以选择跳过或使用更宽松的参数
                    if self.config.skip_low_confidence:
                        # 如果配置了跳过低置信度块
                        self.logger.info(f"Skipping low confidence block {block_idx + 1} due to skip_low_confidence setting")
                        
                        return {
                            'block_idx': block_idx,
                            'success': True,
                            'qa_count': 0,
                            'qa_pairs': [],
                            'skipped': True,
                            'reason': 'Low confidence block skipped by configuration'
                        }
                
                response = self.llm_client.call_ollama(
                    prompt, 
                    temperature=min(0.3, self.config.temperature + 0.05)  # 稍微提高温度，更宽松
                )
                
                if response is None:
                    self.logger.warning(f"❌ Block {block_idx + 1}: LLM call failed")
                    return {
                        'block_idx': block_idx,
                        'success': False,
                        'error': 'LLM call failed',
                        'qa_count': 0
                    }
                
                qa_pairs = self.qa_extractor.extract_json(response)
                
                # 对于低置信度块，即使没有提取到QA对也不算错误
                if not qa_pairs:
                    self.logger.debug(f"No Q&A pairs extracted from low confidence block {block_idx + 1}")
                    return {
                        'block_idx': block_idx,
                        'success': True,
                        'qa_count': 0,
                        'qa_pairs': [],
                        'reason': 'No QA pairs in low confidence block'
                    }
            
            # Process Q&A pairs (包括长答案处理)
            processed_pairs = []
            for qa_pair in qa_pairs:
                # 清理问题文本
                clean_question = self.text_processor.clean_question_text(qa_pair["question"])
                answer = qa_pair["answer"]
                
                # 处理超长答案
                if len(answer) > self.qa_extractor.chain_summary_threshold:
                    self.logger.info(f"Processing long answer for block {block_idx + 1}")
                    answer = self.qa_extractor._process_long_answer(answer, self.llm_client)
                
                processed_pairs.append({
                    "question": clean_question,
                    "answer": answer,
                    "source_text": block_content,
                    "source_confidence": confidence,
                    "source_type": block_type,
                    "domain": domain
                })
            
            # Add metadata to Q&A pairs
            for pair in processed_pairs:
                # Add sliding context if enabled
                if sliding_context:
                    pair["sliding_context"] = sliding_context
                
                # Generate topic for each Q&A pair if enabled
                if enable_llm_anchor:
                    qa_topic = self._generate_qa_topic(pair["question"], pair["answer"])
                    if qa_topic:
                        pair["topic"] = qa_topic
                        self.logger.debug(f"Generated topic for Q&A: {qa_topic}")
            
            # Log to success logger if enabled
            if self.success_logger:
                for i, pair in enumerate(processed_pairs):
                    success_log_content = (
                        f"Successfully extracted Q&A pair from block {block_idx + 1}:\n\n"
                        f"Question: {pair['question']}\n\n"
                        f"Answer: {pair['answer']}\n\n"
                        f"Source block:\n{block_content}\n\n"
                        f"{'='*80}"
                    )
                    self.success_logger.info(success_log_content)
            
            return {
                'block_idx': block_idx,
                'success': True,
                'qa_count': len(processed_pairs),
                'qa_pairs': processed_pairs
            }
            
        except Exception as e:
            self.logger.error(f"❌ Block {block_idx + 1}: Unexpected error: {e}")
            if self.error_logger:
                self.error_logger.error(
                    f"Unexpected error in block {block_idx + 1}: {e}\n"
                    f"Block content:\n{block_data.get('content', 'N/A')}"
                )
            
            return {
                'block_idx': block_idx,
                'success': False,
                'error': f'Unexpected error: {e}',
                'qa_count': 0
            }
    
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
            
            # 使用LLM生成主题关键词
            prompt = f"""请为以下问答对提取 3 个核心关键词。
请只返回关键词本身，并用逗号分隔，不要添加任何其他解释或前缀。

{qa_content[:1500]}

关键词："""
            
            keywords = self.llm_client.call_ollama(prompt, temperature=0.0)
            
            if keywords:
                cleaned_keywords = keywords.strip().replace("关键词：", "").replace("核心关键词：", "").strip()
                self.logger.debug(f"Generated topic for Q&A pair: {cleaned_keywords}")
                return cleaned_keywords
            else:
                self.logger.warning("Failed to generate topic for Q&A pair")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error generating topic for Q&A pair: {e}")
            return ""
    
    def _log_token_monitoring_summary(self):
        """在处理完成后输出token使用总结"""
        if self.token_usage_stats:
            stats = self.token_usage_stats[-1]
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
            self.token_usage_stats[-1]['token_usage'].append(token_count)
            self.token_usage_stats[-1]['max_token_usage'] = max(self.token_usage_stats[-1]['max_token_usage'], token_count)
            self.token_usage_stats[-1]['min_token_usage'] = min(self.token_usage_stats[-1]['min_token_usage'], token_count)
            self.token_usage_stats[-1]['total_blocks_processed'] += 1
            
            # 判断使用的prompt类型
            if self.qa_extractor.compact_prompt in prompt:
                self.token_usage_stats[-1]['prompt_uses']['compact'] += 1
            else:
                self.token_usage_stats[-1]['prompt_uses']['full'] += 1
            
            # 检查是否可能发生截断
            if token_count > self.config.max_prompt_tokens:
                self.token_usage_stats[-1]['truncations'] += 1
                self.logger.warning(f"⚠️ Potential truncation detected: {token_count} tokens > {self.config.max_prompt_tokens} limit")
            
            # 详细日志记录
            self.logger.debug(f"Block token usage: {token_count}/{self.config.max_prompt_tokens} tokens ({token_count/self.config.max_prompt_tokens*100:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Error tracking token usage: {e}")
    
    def _analyze_confidence_processing(self, results: List[Dict[str, Any]], 
                                     processed_blocks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析置信度分级处理的效果"""
        confidence_stats = {
            'high_confidence_blocks': 0,
            'medium_confidence_blocks': 0, 
            'low_confidence_blocks': 0,
            'skipped_blocks': 0,
            'high_confidence_qa_pairs': 0,
            'medium_confidence_qa_pairs': 0,
            'low_confidence_qa_pairs': 0,
            'llm_calls_saved': 0  # 高置信度块节省的LLM调用次数
        }
        
        # 统计原始块的置信度分布
        for block in processed_blocks_data:
            confidence = block.get('confidence', 'unknown')
            if confidence == 'high':
                confidence_stats['high_confidence_blocks'] += 1
                confidence_stats['llm_calls_saved'] += 1  # 高置信度块节省了LLM调用
            elif confidence == 'medium':
                confidence_stats['medium_confidence_blocks'] += 1
            elif confidence == 'low':
                confidence_stats['low_confidence_blocks'] += 1
        
        # 统计处理结果
        for result in results:
            if result.get('skipped'):
                confidence_stats['skipped_blocks'] += 1
            elif result.get('success') and 'qa_pairs' in result:
                # 根据qa_pairs中的source_confidence统计
                for pair in result['qa_pairs']:
                    source_confidence = pair.get('source_confidence', 'unknown')
                    if source_confidence == 'high':
                        confidence_stats['high_confidence_qa_pairs'] += 1
                    elif source_confidence == 'medium':
                        confidence_stats['medium_confidence_qa_pairs'] += 1
                    elif source_confidence == 'low':
                        confidence_stats['low_confidence_qa_pairs'] += 1
        
        return confidence_stats