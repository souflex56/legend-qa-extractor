#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legend QA Extractor - Command Line Interface

A professional Q&A pair extraction tool from PDF documents using local LLMs.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config, load_config, save_config, get_default_config_path
from src.processor import QAExtractionProcessor
from src.utils import setup_logger


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract Q&A pairs from PDF documents using local LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf                          # Extract from document.pdf with default settings
  %(prog)s document.pdf -o output.jsonl         # Specify output file
  %(prog)s document.pdf --config config.yaml    # Use custom configuration
  %(prog)s document.pdf --model qwen2.5:14b     # Use different model
  %(prog)s document.pdf --sample 0.1            # Process only 10%% of blocks for testing
  %(prog)s --validate                           # Validate setup without processing
  %(prog)s --create-config                      # Create sample configuration file
        """
    )
    
    # Input file
    parser.add_argument(
        'pdf_file',
        nargs='?',
        help='Path to PDF file to process'
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create a sample configuration file and exit'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSONL file path'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results and logs'
    )
    
    # Model options
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Ollama model name (e.g., qwen2.5:7b-instruct)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='http://localhost:11434',
        help='Ollama server host (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        help='Model temperature (0.0 to 1.0)'
    )
    
    # Processing options
    parser.add_argument(
        '--max-block-size',
        type=int,
        help='Maximum block size in characters'
    )
    
    parser.add_argument(
        '--min-block-size',
        type=int,
        help='Minimum block size in characters'
    )
    
    parser.add_argument(
        '--sample',
        type=float,
        help='Sample ratio (0.0 to 1.0) for processing subset of blocks'
    )
    
    parser.add_argument(
        '--enable-qa-filter',
        action='store_true',
        help='Enable Q&A filtering (only process blocks with Q&A patterns)'
    )
    
    parser.add_argument(
        '--disable-qa-filter',
        action='store_true',
        help='Disable Q&A filtering (process all blocks)'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (minimal output)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose mode (detailed output)'
    )
    
    # Utility options
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate setup and configuration without processing'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser


def create_sample_config() -> None:
    """Create a sample configuration file."""
    config = Config()
    config_path = 'config/config.yaml'
    
    # Ensure config directory exists
    os.makedirs('config', exist_ok=True)
    
    save_config(config, config_path)
    print(f"✅ Sample configuration created: {config_path}")
    print("You can edit this file to customize your settings.")


def load_and_merge_config(args: argparse.Namespace) -> Config:
    """Load configuration and merge with command line arguments."""
    # Load base configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Try to load default config
        default_config_path = get_default_config_path()
        if os.path.exists(default_config_path):
            config = load_config(default_config_path)
        else:
            config = Config()
    
    # Override with command line arguments
    if args.pdf_file:
        config.pdf_filename = args.pdf_file
    
    if args.output:
        config.output_filename = args.output
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.model:
        config.model_name = args.model
    
    if args.host:
        config.ollama_host = args.host
    
    if args.temperature is not None:
        config.temperature = args.temperature
    
    if args.max_block_size:
        config.max_block_size = args.max_block_size
    
    if args.min_block_size:
        config.min_block_size = args.min_block_size
    
    if args.sample is not None:
        config.extract_ratio = args.sample
    
    if args.enable_qa_filter:
        config.enable_qa_filter = True
    
    if args.disable_qa_filter:
        config.enable_qa_filter = False
    
    if args.log_level:
        config.log_level = args.log_level
    
    # Handle quiet/verbose modes
    if args.quiet:
        config.log_level = 'WARNING'
    elif args.verbose:
        config.log_level = 'DEBUG'
    
    return config


def validate_setup(processor: QAExtractionProcessor) -> bool:
    """Validate the setup and print results."""
    print("🔍 Validating setup...")
    
    validation = processor.validate_setup()
    
    if validation['issues']:
        print("❌ Setup validation failed:")
        for issue in validation['issues']:
            print(f"  • {issue}")
    
    if validation['warnings']:
        print("⚠️ Warnings:")
        for warning in validation['warnings']:
            print(f"  • {warning}")
    
    if validation['valid']:
        print("✅ Setup validation passed!")
        return True
    else:
        print("❌ Setup validation failed. Please fix the issues above.")
        return False


def print_results(results: dict) -> None:
    """Print processing results in a formatted way."""
    if not results['success']:
        print(f"❌ Processing failed: {results['message']}")
        return
    
    stats = results['stats']
    
    print("\n" + "="*50)
    print("📊 PROCESSING RESULTS")
    print("="*50)
    
    print(f"📁 Output file: {results['output_path']}")
    print(f"📄 PDF pages: {stats['pdf_pages']}")
    print(f"📦 Total blocks: {stats['total_blocks']}")
    print(f"✅ Successful blocks: {stats['successful_blocks']}")
    print(f"❌ Failed blocks: {stats['failed_blocks']}")
    print(f"📈 Success rate: {stats['success_rate']:.1%}")
    print(f"🎯 Q&A pairs extracted: {stats['qa_pairs_extracted']}")
    print(f"📊 Average Q&A per block: {stats['avg_qa_per_block']:.1f}")
    
    if stats['quality_metrics']:
        qm = stats['quality_metrics']
        print(f"\n📏 QUALITY METRICS")
        print(f"Question length: {qm['avg_question_length']:.1f} chars (avg)")
        print(f"Answer length: {qm['avg_answer_length']:.1f} chars (avg)")
    
    print(f"\n⚙️ CONFIGURATION USED")
    config_used = stats['config_used']
    print(f"Model: {config_used['model_name']}")
    print(f"Block size: {config_used['min_block_size']}-{config_used['max_block_size']} chars")
    print(f"Extract ratio: {config_used['extract_ratio']:.1%}")
    print(f"QA filter: {'Enabled' if config_used['enable_qa_filter'] else 'Disabled'}")
    print(f"Temperature: {config_used['temperature']}")
    
    print("="*50)


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle special actions
    if args.create_config:
        create_sample_config()
        return
    
    # Load configuration
    try:
        config = load_and_merge_config(args)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Validate required arguments
    if not args.validate and not args.pdf_file:
        parser.error("PDF file is required unless using --validate or --create-config")
    
    # Initialize processor
    try:
        processor = QAExtractionProcessor(config)
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Handle validation mode
    if args.validate:
        if validate_setup(processor):
            print("🎉 System is ready for processing!")
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Validate setup before processing
    if not validate_setup(processor):
        print("Fix the issues above before processing.")
        sys.exit(1)
    
    # Process the PDF
    try:
        print(f"🚀 Starting Q&A extraction from: {args.pdf_file}")
        results = processor.process_pdf(args.pdf_file)
        print_results(results)
        
        if results['success']:
            print("🎉 Processing completed successfully!")
        else:
            print("❌ Processing failed.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 