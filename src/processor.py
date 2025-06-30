"""Main processor class that orchestrates the Q&A extraction workflow."""

import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .config import Config
from .core import PDFProcessor, TextProcessor, QAExtractor, LLMClient
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
        self.qa_extractor = QAExtractor()
        
        # Initialize LLM client
        try:
            self.llm_client = LLMClient(
                host=config.ollama_host,
                model_name=config.model_name
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise
        
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
        self.logger.info("✂️ Creating text blocks...")
        blocks = self.text_processor.create_hybrid_blocks(
            raw_text, 
            self.config.max_block_size, 
            self.config.min_block_size
        )
        self.logger.info(f"✅ Created {len(blocks)} text blocks")
        
        # Filter blocks if QA filtering is enabled
        if self.config.enable_qa_filter:
            original_count = len(blocks)
            blocks = [b for b in blocks if self.text_processor.block_has_qa(b)]
            self.logger.info(f"⚡ QA filtering: {len(blocks)} blocks remaining (from {original_count})")
        
        # Apply sampling ratio
        if self.config.extract_ratio < 1.0:
            blocks = self.text_processor.filter_blocks_by_ratio(blocks, self.config.extract_ratio)
            self.logger.info(f"⚡ Applied sampling ratio: {len(blocks)} blocks selected")
        
        if not blocks:
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
        self.logger.info(f"🤖 Processing {len(blocks)} blocks with LLM...")
        results = self._process_blocks(blocks, output_path)
        
        # Generate final statistics
        stats = self._generate_statistics(results, pdf_info, len(blocks))
        
        self.logger.info(f"🎉 Processing completed! Extracted {stats['qa_pairs_extracted']} Q&A pairs")
        self.logger.info(f"📁 Output saved to: {output_path}")
        
        return {
            'success': True,
            'output_path': output_path,
            'stats': stats,
            'pdf_info': pdf_info
        }
    
    def _process_blocks(self, blocks: List[str], output_path: str) -> List[Dict[str, Any]]:
        """Process text blocks and extract Q&A pairs.
        
        Args:
            blocks: List of text blocks to process
            output_path: Path to save extracted Q&A pairs
            
        Returns:
            List of processing results for each block
        """
        results = []
        success_count = 0
        
        for block_idx, block in enumerate(tqdm(blocks, desc="Extracting Q&A pairs")):
            # Preprocess text
            processed_block = self.text_processor.preprocess_qa_text(block)
            
            # Create prompt
            prompt = self.qa_extractor.create_prompt(processed_block)
            
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
                        f"Block content:\n{block}"
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
                        f"Block content:\n{block}"
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
                qa_pairs, block, self.text_processor
            )
            
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
                        f"Source block:\n{block}\n\n"
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