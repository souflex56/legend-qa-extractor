"""Main processor class that orchestrates the Q&A extraction workflow."""

import os
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from tqdm import tqdm

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
        
        # Initialize LLM client
        try:
            self.llm_client = LLMClient(
                host=config.ollama_host,
                model_name=config.model_name
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise

        self.qa_extractor = QAExtractor(
            llm_client=self.llm_client,
            max_prompt_tokens=config.max_prompt_tokens
        )
        
        # SemanticGrouper for intelligent block creation
        self.semantic_grouper = SemanticGrouper(config)
        
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
        self.logger.info("✂️ Starting text block generation using SemanticGrouper...")
        
        # 预处理文本
        preprocessed_text = self.text_processor.preprocess_qa_text(raw_text)
        paragraphs = [p.strip() for p in preprocessed_text.split('\n\n') if p.strip()]

        # === 核心流程改造 ===
        
        # 1. 规则预筛选 (可选，或融入grouper的第一步)
        # high_confidence_blocks = [p for p in paragraphs if self.text_processor.block_has_qa(p)]

        # 2. 语义动态分组 (核心步骤)
        # 将所有段落送入新的分组器
        semantic_groups = self.semantic_grouper.group(paragraphs)
        
        self.logger.info(f"✅ Generated {len(semantic_groups)} semantic groups for LLM processing.")
        
        # --- BEGIN: Added for block size inspection ---
        self.logger.info("🔍 Individual Group Sizes:")
        total_chars = 0
        for i, group_data in enumerate(semantic_groups):
            size = len(group_data.get('content', ''))
            total_chars += size
            self.logger.info(f"  - Group {i+1}/{len(semantic_groups)}: {size} characters")
        if semantic_groups:
            avg_size = total_chars / len(semantic_groups)
            self.logger.info(f"  - Average group size: {avg_size:.0f} characters")
        # --- END: Added for block size inspection ---
        
        # Filter blocks if QA filtering is enabled
        if self.config.enable_qa_filter:
            original_count = len(semantic_groups)
            semantic_groups = [g for g in semantic_groups if self.text_processor.block_has_qa(g["content"])]
            self.logger.info(f"⚡ QA filtering: {len(semantic_groups)} groups remaining (from {original_count})")
        
        # Apply sampling ratio
        if self.config.extract_ratio < 1.0:
            sample_size = max(int(len(semantic_groups) * self.config.extract_ratio), 1)
            semantic_groups = semantic_groups[:sample_size]
            self.logger.info(f"⚡ Applied sampling ratio: {len(semantic_groups)} groups selected")
        
        if not semantic_groups:
            self.logger.warning("⚠️ No valid semantic groups found for processing")
            return {
                'success': False,
                'message': 'No valid semantic groups found for processing',
                'stats': {'total_blocks': 0, 'qa_pairs_extracted': 0}
            }
        
        # Prepare output
        output_path = self._get_output_path()
        ensure_dir(os.path.dirname(output_path))
        
        # Clear output file
        with open(output_path, "w", encoding="utf-8") as f:
            pass
        
        # Process blocks and extract Q&A pairs
        self.logger.info(f"🤖 Processing {len(semantic_groups)} groups with LLM...")
        results = self.qa_extractor.process_groups(semantic_groups)
        
        # Generate final statistics
        stats = self._generate_statistics(results, pdf_info, len(semantic_groups))
        
        self.logger.info(f"🎉 Processing completed! Extracted {stats['qa_pairs_extracted']} Q&A pairs")
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
    
    def _get_output_path(self) -> str:
        """Constructs the full path for the output file."""
        return os.path.join(self.config.output_dir, self.config.output_filename)

    def _generate_statistics(self, results: List[Dict[str, Any]], 
                           pdf_info: Dict[str, Any],
                           total_blocks: int) -> Dict[str, Any]:
        """Generate final processing statistics.
        
        Args:
            results: List of processing results
            pdf_info: Dictionary with PDF metadata
            total_blocks: Total number of blocks processed
            
        Returns:
            Dictionary with final statistics
        """
        total_qa_pairs = sum(len(r.get('qa_pairs', [])) for r in results if r and r.get('status') == 'success')
        successful_blocks = sum(1 for r in results if r and r.get('status') == 'success')
        failed_blocks = total_blocks - successful_blocks
        
        # Calculate success rate
        success_rate = successful_blocks / total_blocks if total_blocks > 0 else 0
        
        # Calculate average Q&A per block
        avg_qa_per_block = total_qa_pairs / successful_blocks if successful_blocks > 0 else 0
        
        # Calculate quality metrics
        quality_metrics = None
        if total_qa_pairs > 0:
            all_questions = []
            all_answers = []
            for result in results:
                if result and result.get('status') == 'success':
                    for qa_pair in result.get('qa_pairs', []):
                        all_questions.append(qa_pair.get('question', ''))
                        all_answers.append(qa_pair.get('answer', ''))
            
            if all_questions and all_answers:
                quality_metrics = {
                    'avg_question_length': sum(len(q) for q in all_questions) / len(all_questions),
                    'avg_answer_length': sum(len(a) for a in all_answers) / len(all_answers)
                }
        
        stats = {
            'pdf_pages': pdf_info.get('page_count', 0),  # Fixed key name
            'total_blocks': total_blocks,
            'successful_blocks': successful_blocks,
            'failed_blocks': failed_blocks,
            'success_rate': success_rate,  # Added
            'qa_pairs_extracted': total_qa_pairs,
            'avg_qa_per_block': avg_qa_per_block,  # Added
            'quality_metrics': quality_metrics,  # Added
            'config_used': {  # Added configuration information
                'model_name': self.config.model_name,
                'min_block_size': self.config.min_block_size,
                'max_block_size': self.config.max_block_size,
                'extract_ratio': self.config.extract_ratio,
                'enable_qa_filter': self.config.enable_qa_filter,
                'temperature': self.config.temperature
            }
        }
        
        # Add token usage stats if available
        if self.config.enable_token_monitoring and self.token_usage_stats:
            summary = self.token_usage_stats[-1]
            stats.update({
                'total_prompt_tokens': sum(u['tokens'] for u in summary['token_usage']),
                'max_prompt_tokens_used': summary['max_token_usage'],
                'min_prompt_tokens_used': summary['min_token_usage'],
                'prompts_truncated': summary['truncations'],
                'compact_prompts_used': summary['prompt_uses']['compact'],
                'full_prompts_used': summary['prompt_uses']['full']
            })
            
        return stats
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate the environment and configuration.
        
        Returns:
            A dictionary with 'valid', 'issues', and 'warnings' keys.
        """
        self.logger.info("⚙️ Validating setup...")
        
        issues = []
        warnings = []
        
        # 1. Validate PDF file existence
        try:
            pdf_path = self.config.pdf_filename
            if not os.path.isabs(pdf_path):
                # Try to resolve relative path
                resolved_path = os.path.abspath(pdf_path)
                if not os.path.exists(resolved_path):
                    # Check in common data directories if needed
                    # For now, just check the resolved path
                    pass # Keep pdf_path as is for error message
            else:
                resolved_path = pdf_path
            
            if not os.path.exists(resolved_path):
                issues.append(f"PDF file not found at '{pdf_path}'")
        except Exception as e:
            issues.append(f"Invalid PDF path configuration: {e}")
            
        # 2. Validate Ollama connection and model availability
        try:
            if not self.llm_client.check_model_availability():
                issues.append(f"Model '{self.config.model_name}' not available on Ollama server at {self.config.ollama_host}")
        except Exception as e:
            issues.append(f"Failed to connect to Ollama at {self.config.ollama_host}: {e}")
        
        # 3. Validate output directory
        try:
            ensure_dir(self.config.output_dir)
        except Exception as e:
            issues.append(f"Could not create or access output directory '{self.config.output_dir}': {e}")

        # 4. Check for potential performance issues (as warnings)
        if self.config.max_block_size > 4000:
            warnings.append(f"max_block_size ({self.config.max_block_size}) is large, which may slow down processing.")
        
        if self.config.temperature > 0.7:
            warnings.append(f"Model temperature ({self.config.temperature}) is high, which may lead to less factual Q&A pairs.")

        return {
            'valid': not issues,
            'issues': issues,
            'warnings': warnings
        }

    def _generate_qa_topic(self, question: str, answer: str) -> str:
        """Generate a concise topic for a given Q&A pair using LLM.
        
        Args:
            question: The question text
            answer: The answer text
            
        Returns:
            A string representing the topic, or "general" if generation fails
        """
        try:
            topic = self.qa_extractor.generate_topic(question, answer)
            return topic
        except Exception as e:
            self.logger.warning(f"⚠️ Topic generation failed: {e}. Defaulting to 'general'.")
            return "general"

    def _log_token_monitoring_summary(self):
        """Logs a summary of token usage if monitoring is enabled."""
        if not self.config.enable_token_monitoring or not self.token_usage_stats:
            return

        summary = self.token_usage_stats[-1]
        total_tokens = sum(u['tokens'] for u in summary['token_usage'])
        avg_tokens = total_tokens / summary['total_blocks_processed'] if summary['total_blocks_processed'] > 0 else 0
        
        self.logger.info("--- 📊 Token Usage Summary ---")
        self.logger.info(f"Total Blocks Processed: {summary['total_blocks_processed']}")
        self.logger.info(f"Total Prompt Tokens Used: {total_tokens}")
        self.logger.info(f"Average Tokens per Block: {avg_tokens:.2f}")
        self.logger.info(f"Max Tokens in a Single Prompt: {summary['max_token_usage']}")
        self.logger.info(f"Min Tokens in a Single Prompt: {summary['min_token_usage']}")
        self.logger.info(f"Number of Truncated Prompts: {summary['truncations']}")
        self.logger.info(f"Compact Prompts Used (Context only): {summary['prompt_uses']['compact']}")
        self.logger.info(f"Full Prompts Used (Context + Anchor): {summary['prompt_uses']['full']}")
        self.logger.info("-----------------------------")

    def _track_token_usage(self, prompt: str, block_anchor: str, sliding_context: str):
        """Tracks token usage for a single block processing event."""
        full_prompt = self.qa_extractor.get_full_prompt(prompt, block_anchor, sliding_context)
        token_count = self.llm_client.count_tokens(full_prompt)
        
        summary = self.token_usage_stats[-1]
        summary['token_usage'].append({'tokens': token_count, 'truncated': token_count > self.config.max_prompt_tokens})
        
        if token_count > summary['max_token_usage']:
            summary['max_token_usage'] = token_count
        if token_count < summary['min_token_usage']:
            summary['min_token_usage'] = token_count
        if token_count > self.config.max_prompt_tokens:
            summary['truncations'] += 1
            
        if block_anchor:
            summary['prompt_uses']['full'] += 1
        else:
            summary['prompt_uses']['compact'] += 1
            
        summary['total_blocks_processed'] += 1