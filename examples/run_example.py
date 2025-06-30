#!/usr/bin/env python3
"""
Example script showing how to use Legend QA Extractor programmatically.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.config import Config, load_config, save_config
from src.processor import QAExtractionProcessor


def run_basic_example():
    """Run a basic example with default configuration."""
    print("🚀 Running basic example...")
    
    # Create default configuration
    config = Config()
    config.pdf_filename = "sample_document.pdf"  # Replace with your PDF
    config.output_filename = "example_output.jsonl"
    config.extract_ratio = 0.1  # Process only 10% for quick testing
    config.log_level = "DEBUG"  # Verbose logging
    
    # Initialize processor
    try:
        processor = QAExtractionProcessor(config)
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return False
    
    # Validate setup
    validation = processor.validate_setup()
    if not validation['valid']:
        print("❌ Setup validation failed:")
        for issue in validation['issues']:
            print(f"  • {issue}")
        return False
    
    # Process PDF
    try:
        results = processor.process_pdf()
        
        if results['success']:
            stats = results['stats']
            print(f"✅ Success! Extracted {stats['qa_pairs_extracted']} Q&A pairs")
            print(f"📁 Output saved to: {results['output_path']}")
            return True
        else:
            print(f"❌ Processing failed: {results['message']}")
            return False
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False


def run_config_file_example():
    """Run example using configuration file."""
    print("🚀 Running config file example...")
    
    # Load configuration from file
    config_path = "examples/sample_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Creating sample config file...")
        
        # Create sample config
        config = Config()
        config.pdf_filename = "your_document.pdf"
        config.extract_ratio = 0.1
        save_config(config, config_path)
        print(f"✅ Sample config created: {config_path}")
        print("Please edit the config file and run again.")
        return False
    
    try:
        config = load_config(config_path)
        processor = QAExtractionProcessor(config)
        
        # Validate and process
        validation = processor.validate_setup()
        if validation['valid']:
            results = processor.process_pdf()
            if results['success']:
                print(f"✅ Success! Check output at: {results['output_path']}")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_custom_settings_example():
    """Run example with custom settings."""
    print("🚀 Running custom settings example...")
    
    # Create configuration with custom settings
    config = Config()
    
    # Custom file settings
    config.pdf_filename = "interview_transcript.pdf"
    config.output_filename = "interview_qa.jsonl"
    config.output_dir = "custom_output"
    
    # Custom processing settings
    config.max_block_size = 2000  # Larger blocks for more context
    config.min_block_size = 50    # Smaller minimum for short Q&As
    config.enable_qa_filter = True  # Only process Q&A blocks
    config.extract_ratio = 0.2    # Process 20% for testing
    
    # Custom model settings
    config.model_name = "qwen2.5:14b-instruct"  # Use larger model if available
    config.temperature = 0.05  # Very consistent responses
    
    # Custom prefixes for interview context
    config.known_prefixes.extend(["面试官", "候选人", "主管", "HR"])
    
    print("Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Block size: {config.min_block_size}-{config.max_block_size}")
    print(f"  QA filter: {config.enable_qa_filter}")
    print(f"  Extract ratio: {config.extract_ratio:.1%}")
    
    try:
        processor = QAExtractionProcessor(config)
        
        # Check if model is available
        if not processor.llm_client.check_model_availability():
            print(f"⚠️ Model {config.model_name} not available, trying to pull...")
            if not processor.llm_client.pull_model():
                print("❌ Failed to pull model. Using default model.")
                config.model_name = "qwen2.5:7b-instruct"
                processor.llm_client.set_model(config.model_name)
        
        validation = processor.validate_setup()
        if validation['valid']:
            results = processor.process_pdf()
            if results['success']:
                print(f"✅ Custom processing completed!")
                print(f"📊 Results: {results['stats']}")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main function to run examples."""
    print("=== Legend QA Extractor Examples ===\n")
    
    examples = [
        ("Basic Example", run_basic_example),
        ("Config File Example", run_config_file_example),
        ("Custom Settings Example", run_custom_settings_example),
    ]
    
    for name, func in examples:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print('='*50)
        
        success = func()
        
        if success:
            print(f"✅ {name} completed successfully!")
        else:
            print(f"❌ {name} failed.")
        
        print()
    
    print("=== Examples completed ===")


if __name__ == "__main__":
    main() 