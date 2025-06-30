#!/usr/bin/env python3
"""Setup script for Legend QA Extractor."""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "A professional Q&A pair extraction tool from PDF documents using local LLMs."

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return ["ollama>=0.1.7", "PyMuPDF>=1.23.0", "tqdm>=4.65.0", "PyYAML>=6.0"]

setup(
    name="legend-qa-extractor",
    version="1.0.0",
    author="LegendQA Team",
    author_email="",
    description="A professional Q&A pair extraction tool from PDF documents using local LLMs",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/legend-qa-extractor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "ui": [
            "streamlit>=1.28.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "legend-qa-extractor=extract_qa:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md", "*.txt"],
    },
    keywords="nlp, question-answering, pdf-extraction, llm, ollama, chinese",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/legend-qa-extractor/issues",
        "Source": "https://github.com/yourusername/legend-qa-extractor",
        "Documentation": "https://github.com/yourusername/legend-qa-extractor/docs",
    },
) 