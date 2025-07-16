"""Main processor class that orchestrates the Q&A extraction workflow."""

import os
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import time
import re # Added for regex in _extract_complete_qa_first

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
        
        # 🔥 重新设计的处理流程：先识别问答对，再分块
        self.logger.info("🔍 Starting QA-first processing pipeline...")
        
        # 预处理文本
        preprocessed_text = self.text_processor.preprocess_qa_text(raw_text)
        
        # 分割成段落
        paragraphs = [p.strip() for p in preprocessed_text.split('\n') if p.strip()]
        self.logger.info(f"📝 Split into {len(paragraphs)} paragraphs")
        
        # 🔥 第一步：全文rule-based扫描，识别完整问答对
        self.logger.info("🎯 Step 1: Rule-based full-text QA identification...")
        complete_qa_blocks, remaining_paragraphs = self._extract_complete_qa_first(paragraphs)
        
        self.logger.info(f"✅ Found {len(complete_qa_blocks)} complete QA blocks")
        self.logger.info(f"📋 Remaining {len(remaining_paragraphs)} paragraphs for semantic grouping")
        
        # 🔥 第二步：对剩余段落进行语义分组
        semantic_blocks = []
        if remaining_paragraphs:
            self.logger.info("🧠 Step 2: Semantic grouping for remaining content...")
            # 获取剩余段落在原始列表中的索引
            remaining_indices = [i for i, para in enumerate(paragraphs) if para in remaining_paragraphs]
            semantic_blocks = self.semantic_grouper._semantic_dynamic_grouping(paragraphs, remaining_indices)
            self.logger.info(f"✅ Created {len(semantic_blocks)} semantic blocks")
        
        # 🔥 第三步：合并所有块
        all_blocks = complete_qa_blocks + semantic_blocks
        self.logger.info(f"📦 Total blocks for processing: {len(all_blocks)}")
        
        # --- BEGIN: Added for block size inspection ---
        self.logger.info("🔍 Block Analysis:")
        total_chars = 0
        qa_complete_count = sum(1 for b in all_blocks if b.get('qa_complete', False))
        for i, block_data in enumerate(all_blocks):
            size = len(block_data.get('content', ''))
            total_chars += size
            block_type = "QA-Complete" if block_data.get('qa_complete', False) else "Semantic"
            self.logger.info(f"  - Block {i+1}/{len(all_blocks)} [{block_type}]: {size} characters")
        if all_blocks:
            avg_size = total_chars / len(all_blocks)
            self.logger.info(f"  - Complete QA blocks: {qa_complete_count}")
            self.logger.info(f"  - Semantic blocks: {len(all_blocks) - qa_complete_count}")
            self.logger.info(f"  - Average block size: {avg_size:.0f} characters")
        # --- END: Added for block size inspection ---
        
        # Filter blocks if QA filtering is enabled
        if self.config.enable_qa_filter:
            original_count = len(all_blocks)
            all_blocks = [b for b in all_blocks if b.get('qa_complete', False) or self.text_processor.block_has_qa(b["content"])]
            self.logger.info(f"⚡ QA filtering: {len(all_blocks)} blocks remaining (from {original_count})")
        
        # Apply sampling ratio
        if self.config.extract_ratio < 1.0:
            sample_size = max(int(len(all_blocks) * self.config.extract_ratio), 1)
            
            # 根据采样策略选择blocks
            if getattr(self.config, 'sampling_strategy', 'sequential') == 'random':
                import random
                # 随机采样
                all_blocks = random.sample(all_blocks, sample_size)
                self.logger.info(f"⚡ Applied random sampling ratio: {len(all_blocks)} blocks selected")
            else:
                # 顺序采样（从头开始）
                all_blocks = all_blocks[:sample_size]
                self.logger.info(f"⚡ Applied sequential sampling ratio: {len(all_blocks)} blocks selected")
        
        if not all_blocks:
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
        self.logger.info(f"🤖 Processing {len(all_blocks)} blocks with optimized pipeline...")
        results = self._process_blocks(all_blocks, output_path, self.config.enable_llm_anchor)
        
        # Generate final statistics with confidence analysis
        stats = self._generate_statistics(results, pdf_info, len(all_blocks))
        
        # Add confidence-based processing statistics
        confidence_stats = self._analyze_confidence_processing(results, all_blocks)
        
        # 🔥 增强日志输出，显示新处理流程的效果
        self.logger.info(f"🎉 QA-First Processing Pipeline Completed!")
        self.logger.info(f"📊 Processing Summary:")
        self.logger.info(f"   - Total QA pairs extracted: {stats['qa_pairs_extracted']}")
        self.logger.info(f"   - Complete QA blocks: {qa_complete_count} (direct rule-based)")
        self.logger.info(f"   - Semantic blocks: {len(all_blocks) - qa_complete_count} (LLM processed)")
        self.logger.info(f"   - Processing efficiency: {len(all_blocks)} blocks → {stats['qa_pairs_extracted']} QA pairs")
        
        # 输出方法统计
        method_stats = {}
        for result in results:
            if result.get('success', False):
                method = result.get('method', 'unknown')
                method_stats[method] = method_stats.get(method, 0) + result.get('qa_count', 0)
        
        if method_stats:
            self.logger.info(f"📋 Extraction Methods Used:")
            for method, count in method_stats.items():
                self.logger.info(f"   - {method}: {count} QA pairs")
        
        self.logger.info(f"📁 Output saved to: {output_path}")
        
        # 输出token监控总结（如果启用）
        if self.config.enable_token_monitoring:
            self._log_token_monitoring_summary()
        
        # **🚀 PERFORMANCE OPTIMIZATION: Log performance summary**
        self.llm_client.log_performance_summary()
        
        # Success response
        return {
            'success': True,
            'output_path': output_path,
            'stats': stats,
            'confidence_stats': confidence_stats,
            'processing_summary': {
                'complete_qa_blocks': qa_complete_count,
                'semantic_blocks': len(all_blocks) - qa_complete_count,
                'total_blocks': len(all_blocks),
                'method_stats': method_stats
            }
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
            confidence = block_data.get("confidence", "unknown")
            block_type = block_data.get("type", "unknown")
            domain = block_data.get("domain", "general")
            qa_complete = block_data.get("qa_complete", False)
            
            # 🔥 新增：对于完整QA块，直接进行rule-based提取
            if qa_complete:
                self.logger.debug(f"Processing complete QA block {block_idx + 1} with direct rule-based extraction")
                
                # 直接使用QA提取器的规则方法
                qa_pairs = self.qa_extractor._extract_from_high_confidence_block(block_content)
                
                # 处理提取结果
                processed_pairs = []
                for qa_pair in qa_pairs:
                    clean_question = self.text_processor.clean_question_text(qa_pair["question"])
                    answer = qa_pair["answer"]
                    
                    # 处理超长答案
                    if len(answer) > self.qa_extractor.chain_summary_threshold:
                        self.logger.info(f"Processing long answer for complete QA block {block_idx + 1}")
                        answer = self.qa_extractor._process_long_answer(answer, self.llm_client)
                    
                    processed_pairs.append({
                        "question": clean_question,
                        "answer": answer,
                        "source_text": block_content,
                        "source_confidence": "high",  # 完整QA块总是高置信度
                        "source_type": "complete_qa_pair",
                        "domain": domain
                    })
                
                processing_time = time.time() - start_time
                
                return {
                    'block_idx': block_idx,
                    'success': True,
                    'qa_count': len(processed_pairs),
                    'qa_pairs': processed_pairs,
                    'processing_time': processing_time,
                    'method': 'complete_qa_direct'
                }
            
            # 🔥 对于非完整QA块，使用原有的置信度处理逻辑
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
                
                # 对于中置信度块，至少尝试一次rule-based作为fallback
                if not qa_pairs:
                    self.logger.debug(f"LLM extraction failed for medium confidence block {block_idx + 1}, trying rule-based fallback")
                    qa_pairs = self.qa_extractor._extract_from_high_confidence_block(processed_block)
                
            else:
                # 低置信度块：使用宽松LLM处理
                self.logger.debug(f"Processing {confidence} confidence block {block_idx + 1} with LLM (permissive)")
                
                # 为低置信度块适当增加温度，可能发现更多隐藏的QA对
                response = self.llm_client.call_ollama(
                    prompt, 
                    temperature=min(0.3, self.config.temperature + 0.05)  # 稍微增加温度
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
            
            # Add LLM anchors if enabled
            if enable_llm_anchor and processed_pairs:
                self.logger.debug(f"Generating LLM anchor for block {block_idx + 1}")
                anchor = self._generate_llm_anchor(block_content)
                if anchor:
                    for pair in processed_pairs:
                        pair["llm_anchor"] = anchor
            
            processing_time = time.time() - start_time
            
            # Log success
            if self.success_logger and processed_pairs:
                for pair in processed_pairs:
                    self.success_logger.info(
                        f"Block {block_idx + 1} - Q: {pair['question'][:100]}...\n"
                        f"A: {pair['answer'][:200]}..."
                    )
            
            return {
                'block_idx': block_idx,
                'success': True,
                'qa_count': len(processed_pairs),
                'qa_pairs': processed_pairs,
                'processing_time': processing_time,
                'method': 'complete_qa_direct' if qa_complete else f'{confidence}_confidence_llm'
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing block {block_idx + 1}: {e}"
            self.logger.error(error_msg)
            
            if self.error_logger:
                self.error_logger.error(
                    f"{error_msg}\n"
                    f"Block content:\n{block_content}"
                )
            
            return {
                'block_idx': block_idx,
                'success': False,
                'error': str(e),
                'qa_count': 0,
                'processing_time': processing_time
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

    def _extract_complete_qa_first(self, paragraphs: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        🔥 核心方法：全文rule-based扫描，优先识别完整问答对
        
        这是新设计的核心：不再受块大小限制，直接在全文范围内识别完整问答对
        
        Args:
            paragraphs: 所有段落列表
            
        Returns:
            Tuple[完整QA块列表, 剩余未配对段落列表]
        """
        self.logger.info(f"🔍 Scanning {len(paragraphs)} paragraphs for complete QA pairs...")
        
        # 使用QA提取器的增强问答识别逻辑
        # 定义说话人前缀
        questioner_prefixes = ['网友', '问', 'Q', '记者', '提问', '主持人', '观众']
        answerer_prefixes = ['段永平', '段', '大道', '答', 'A']
        
        # 构建正则表达式
        questioner_patterns = []
        answerer_patterns = []
        
        for prefix in questioner_prefixes:
            questioner_patterns.append(f'{re.escape(prefix)}\\s*[A-Za-z0-9\u4e00-\u9fa5]*')
            questioner_patterns.append(f'[A-Za-z0-9\u4e00-\u9fa5]+\\s+{re.escape(prefix)}')
        
        for prefix in answerer_prefixes:
            answerer_patterns.append(f'{re.escape(prefix)}\\s*[A-Za-z0-9\u4e00-\u9fa5]*')
            answerer_patterns.append(f'[A-Za-z0-9\u4e00-\u9fa5]+\\s+{re.escape(prefix)}')
        
        # 收集所有说话段落
        segments = []
        used_indices = set()
        
        for i, para in enumerate(paragraphs):
            # 检查是否是问题
            is_questioner = self._is_speaker_type(para, questioner_patterns)
            # 检查是否是答案
            is_answerer = self._is_speaker_type(para, answerer_patterns)
            
            if is_questioner or is_answerer:
                segments.append({
                    'index': i,
                    'content': para,
                    'is_questioner': is_questioner,
                    'is_answerer': is_answerer
                })
        
        self.logger.info(f"🎯 Found {len(segments)} speaker segments (Q/A candidates)")
        
        # 智能配对逻辑
        complete_qa_blocks = []
        i = 0
        
        while i < len(segments):
            segment = segments[i]
            
            if segment['is_questioner']:
                # 找到问题，寻找对应答案
                qa_content = [segment['content']]
                qa_indices = [segment['index']]
                used_indices.add(segment['index'])
                
                # 向前寻找答案
                answer_found = False
                j = i + 1
                
                # 在接下来的段落中寻找答案（最多查看5个段落）
                while j < len(segments) and j < i + 6:
                    next_segment = segments[j]
                    
                    # 如果找到答案者
                    if next_segment['is_answerer']:
                        qa_content.append(next_segment['content'])
                        qa_indices.append(next_segment['index'])
                        used_indices.add(next_segment['index'])
                        answer_found = True
                        
                        # 🔥 收集同一回答者的后续补充
                        k = j + 1
                        while k < len(segments) and k < j + 3:
                            follow_up = segments[k]
                            # 检查后续段落
                            if k < len(paragraphs) - 1:
                                next_para = paragraphs[segments[k]['index'] + 1] if segments[k]['index'] + 1 < len(paragraphs) else ""
                                # 如果是同一回答者的继续，或者是无标识的补充内容
                                if (follow_up['is_answerer'] or 
                                    (not follow_up['is_questioner'] and not self._is_speaker_type(next_para, questioner_patterns + answerer_patterns))):
                                    qa_content.append(follow_up['content'])
                                    qa_indices.append(follow_up['index'])
                                    used_indices.add(follow_up['index'])
                                    k += 1
                                else:
                                    break
                            else:
                                break
                        break
                    
                    # 如果遇到新问题，停止
                    elif next_segment['is_questioner']:
                        break
                    
                    j += 1
                
                # 如果找到完整问答对，创建块
                if answer_found and len(qa_content) >= 2:
                    # 收集问答对之间的任何中间段落
                    start_idx = min(qa_indices)
                    end_idx = max(qa_indices)
                    
                    # 添加中间的任何相关段落
                    for idx in range(start_idx + 1, end_idx):
                        if idx not in used_indices and idx < len(paragraphs):
                            # 检查是否是相关的中间内容（短段落且不是新的问答）
                            middle_para = paragraphs[idx]
                            if (len(middle_para) < 100 and 
                                not self._is_speaker_type(middle_para, questioner_patterns + answerer_patterns)):
                                qa_content.insert(-1, middle_para)  # 插在答案前
                                qa_indices.append(idx)
                                used_indices.add(idx)
                    
                    # 创建完整QA块
                    complete_content = "\n\n".join(qa_content)
                    complete_qa_blocks.append({
                        'content': complete_content,
                        'confidence': 'high',
                        'type': 'complete_qa_pair',
                        'qa_complete': True,  # 标记为完整问答对
                        'indices': sorted(qa_indices),
                        'domain': 'general',
                        'qa_count': 1  # 包含一个完整问答对
                    })
                    
                    self.logger.debug(f"✅ Created complete QA block from indices {sorted(qa_indices)}")
            
            i += 1
        
        # 收集剩余未使用的段落
        remaining_paragraphs = [para for i, para in enumerate(paragraphs) if i not in used_indices]
        
        self.logger.info(f"🎉 Successfully identified {len(complete_qa_blocks)} complete QA pairs")
        self.logger.info(f"📋 {len(remaining_paragraphs)} paragraphs remain for semantic processing")
        
        return complete_qa_blocks, remaining_paragraphs
    
    def _is_speaker_type(self, paragraph: str, patterns: List[str]) -> bool:
        """检查段落是否匹配说话人模式"""
        if not paragraph:
            return False
        
        # 移除编号前缀
        clean_para = re.sub(r'^\d+\.\s*', '', paragraph).strip()
        
        # 检查是否匹配任何模式
        for pattern in patterns:
            if re.search(f'(?:^|\\n)\\s*{pattern}\\s*[:：]', clean_para):
                return True
        
        return False