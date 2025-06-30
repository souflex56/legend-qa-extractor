# 🔍 Legend QA Extractor

<div align="right">

**Language** | **语言**: [🇺🇸 EN](README_EN.md) | [🇨🇳 中文](README.md)

</div>

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A professional Q&A pair extraction tool from PDF documents using local LLMs**

*Transform your PDF documents into structured Q&A datasets for AI training*

</div>

## ✨ Key Features

- 🤖 **Local LLM Integration**: Uses Ollama with models like Qwen2.5 for privacy-focused processing
- 📄 **Smart PDF Processing**: Advanced text extraction and intelligent block segmentation
- 🎯 **Intelligent Q&A Detection**: Multi-pattern recognition for various question-answer formats
- ⚙️ **Highly Configurable**: YAML configuration files with environment variable support
- 🔧 **Developer Friendly**: Modular architecture, comprehensive tests, and CLI interface
- 📊 **Quality Metrics**: Built-in extraction quality assessment and detailed logging
- 🚀 **Production Ready**: Type hints, error handling, and professional code structure

## 🔄 How It Works

```
📄 PDF Input → 📝 Text Split → 🔍 Filter Blocks → 🤖 AI Analysis → 📊 Q&A Pairs
                ↑ Block Size      ↑ QA Filter      ↑ Model + Temp
               100-1500 chars    on/off + Sample   qwen2.5:7b/custom
                 (custom)           0.1-1.0           Temp: 0.0-1.0
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- A PDF document to process

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/legend-qa-extractor.git
cd legend-qa-extractor

# Set up environment (creates venv, installs dependencies)
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Or manual setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Basic Usage

```bash
# Extract Q&A pairs from a PDF
python extract_qa.py document.pdf

# Use custom output file
python extract_qa.py document.pdf -o my_output.jsonl

# Process only 10% for testing
python extract_qa.py document.pdf --sample 0.1

# Use different model
python extract_qa.py document.pdf --model qwen2.5:14b-instruct

# Create configuration file
python extract_qa.py --create-config

# Validate setup
python extract_qa.py --validate
```

## 📁 Project Structure

```
legend-qa-extractor/
├── src/                          # Core source code
│   ├── config/                   # Configuration management
│   ├── core/                     # Core processing modules
│   │   ├── pdf_processor.py      # PDF text extraction
│   │   ├── text_processor.py     # Text segmentation & cleaning
│   │   ├── qa_extractor.py       # Q&A pair extraction
│   │   └── llm_client.py         # Ollama integration
│   └── utils/                    # Utility functions
├── config/                       # Configuration files
├── examples/                     # Usage examples
├── tests/                        # Test suite
├── scripts/                      # Setup and utility scripts
├── extract_qa.py                 # CLI entry point
└── output/                       # Generated results
```

## ⚙️ Configuration

### YAML Configuration File

Create a configuration file for your specific needs:

```yaml
# config/config.yaml
pdf_filename: "your_document.pdf"
output_filename: "extracted_qa.jsonl"
model_name: "qwen2.5:7b-instruct"
max_block_size: 1500
extract_ratio: 1.0
enable_qa_filter: false
```

### Environment Variables

Override settings with environment variables:

```bash
export PDF_FILENAME="document.pdf"
export OLLAMA_MODEL="qwen2.5:14b-instruct"
export MAX_BLOCK_SIZE=2000
export EXTRACT_RATIO=0.5
```

### Command Line Options

```bash
python extract_qa.py --help

Options:
  --config CONFIG         YAML configuration file
  --output OUTPUT         Output JSONL file
  --model MODEL          Ollama model name
  --sample RATIO         Sample ratio (0.0-1.0)
  --enable-qa-filter     Only process Q&A blocks
  --temperature TEMP     Model temperature
  --log-level LEVEL      Logging level
  --validate             Validate setup only
```

## 🎯 Advanced Usage

### Programmatic API

```python
from src.config import Config
from src.processor import QAExtractionProcessor

# Create configuration
config = Config()
config.pdf_filename = "document.pdf"
config.model_name = "qwen2.5:7b-instruct"
config.extract_ratio = 0.1  # Quick test

# Initialize processor
processor = QAExtractionProcessor(config)

# Validate setup
validation = processor.validate_setup()
if validation['valid']:
    # Process PDF
    results = processor.process_pdf()
    print(f"Extracted {results['stats']['qa_pairs_extracted']} Q&A pairs")
```

### Custom Configuration

```python
# Custom settings for interview transcripts
config = Config()
config.known_prefixes.extend(["面试官", "候选人", "HR"])
config.max_block_size = 2000
config.enable_qa_filter = True
config.temperature = 0.05
```

## 📊 Output Format

The tool generates JSONL files with structured Q&A pairs:

```json
{
  "question": "What is value investing?",
  "answer": "Value investing is an investment strategy that focuses on the intrinsic value of companies...",
  "source_text": "User: What is value investing?\nExpert: Value investing is an investment strategy..."
}
```

## 🔧 Development

### Setup Development Environment

```bash
# Install with development dependencies
./scripts/setup_environment.sh --dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Code formatting
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_text_processor.py

# Run integration tests
pytest -m integration
```

## 🏗️ Architecture

### Core Components

1. **PDFProcessor**: Extracts text from PDF documents
2. **TextProcessor**: Segments text and applies cleaning
3. **QAExtractor**: Identifies and extracts Q&A pairs using LLM
4. **LLMClient**: Manages communication with Ollama
5. **Config**: Centralized configuration management

## 🎛️ Configuration Options

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `pdf_filename` | PDF file to process | `"uploaded.pdf"` | `"interview.pdf"` |
| `model_name` | Ollama model | `"qwen2.5:7b-instruct"` | `"qwen2.5:14b"` |
| `max_block_size` | Max text block size | `1500` | `2000` |
| `extract_ratio` | Fraction of blocks to process | `1.0` | `0.1` |
| `enable_qa_filter` | Filter blocks with QA patterns | `false` | `true` |
| `temperature` | Model creativity | `0.1` | `0.05` |

## 📈 Performance Tips

1. **Model Selection**: Use `qwen2.5:7b-instruct` for speed, `qwen2.5:14b-instruct` for quality
2. **Block Size**: Larger blocks (2000+) provide more context but slower processing
3. **QA Filtering**: Enable for documents with clear Q&A structure to improve speed
4. **Sampling**: Use `extract_ratio=0.1` for quick testing
5. **Batch Processing**: Process multiple files with different configurations

## 🛠️ Troubleshooting

### Common Issues

**Ollama Connection Failed**
```bash
# Check if Ollama is running
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

**Model Not Found**
```bash
# Pull the required model
ollama pull qwen2.5:7b-instruct
```

**Low Quality Extractions**
- Increase `temperature` for more creative responses
- Adjust `max_block_size` for better context
- Enable `enable_qa_filter` for focused processing

### Debug Mode

```bash
# Enable verbose logging
python extract_qa.py document.pdf --log-level DEBUG

# Check validation
python extract_qa.py --validate
```

## 📚 Examples

See the `examples/` directory for:
- `run_example.py`: Programmatic usage examples
- `sample_config.yaml`: Configuration templates
- Various use case scenarios

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/yourusername/legend-qa-extractor.git
cd legend-qa-extractor
./scripts/setup_environment.sh --dev
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM infrastructure
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing
- The open-source community for inspiration and tools

## 📞 Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/yourusername/legend-qa-extractor/issues)
- 💬 [Discussions](https://github.com/yourusername/legend-qa-extractor/discussions)

---

<div align="center">

**Star ⭐ this repository if you find it helpful!**


</div>


