#!/usr/bin/env python3
"""简单的调试脚本来找出问题所在"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    print("🔍 Testing imports...")
    try:
        from src.config import Config, load_config
        print("✅ Config import OK")
        
        from src.core import PDFProcessor, LLMClient
        print("✅ Core imports OK")
        
        from src.core.semantic_grouper import SemanticGrouper
        print("✅ SemanticGrouper import OK")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    return True

def test_pdf_processing():
    print("\n🔍 Testing PDF processing...")
    try:
        from src.core import PDFProcessor
        processor = PDFProcessor()
        print("✅ PDFProcessor created")
        
        # Test if file exists
        pdf_path = "uploaded.pdf"
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return False
        print(f"✅ PDF file exists: {pdf_path}")
        
        # Try to extract just first few characters
        print("🔍 Extracting text from PDF...")
        text = processor.extract_text_from_pdf(pdf_path)
        print(f"✅ Extracted {len(text)} characters")
        print(f"📄 First 200 chars: {text[:200]}...")
        
    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        return False
    return True

def test_llm_client():
    print("\n🔍 Testing LLM client...")
    try:
        from src.core import LLMClient
        client = LLMClient(host="http://localhost:11434", model_name="qwen2.5:7b-instruct")
        print("✅ LLMClient created")
        
        # Test a simple call
        print("🔍 Testing simple LLM call...")
        response = client.call_ollama("你好", temperature=0.1)
        print(f"✅ LLM response: {response[:100]}...")
        
    except Exception as e:
        print(f"❌ LLM client error: {e}")
        return False
    return True

def test_semantic_grouper():
    print("\n🔍 Testing semantic grouper...")
    try:
        from src.config import Config
        from src.core.semantic_grouper import SemanticGrouper
        
        config = Config()
        print(f"📏 Config - min_block_size: {config.min_block_size}, max_block_size: {config.max_block_size}")
        
        grouper = SemanticGrouper(config)
        print("✅ SemanticGrouper created")
        
        # Test with simple text first
        test_paragraphs = ["这是第一段。", "这是第二段。", "这是第三段。"]
        print(f"🔍 Testing grouping with simple text (lengths: {[len(p) for p in test_paragraphs]})...")
        groups = grouper.group(test_paragraphs)
        print(f"✅ Simple test - Grouped into {len(groups)} groups")
        
        # Test with longer text
        long_paragraphs = [
            "这是一个比较长的段落，包含了更多的内容。段永平是著名的投资人，他有很多投资心得。" * 3,
            "价值投资是一个重要的投资理念，需要长期坚持。网友经常询问各种投资问题。" * 3,
            "关于投资的问题很多，段永平总是耐心回答大家的疑问。这些问答很有价值。" * 3
        ]
        print(f"🔍 Testing grouping with longer text (lengths: {[len(p) for p in long_paragraphs]})...")
        groups = grouper.group(long_paragraphs)
        print(f"✅ Long test - Grouped into {len(groups)} groups")
        if groups:
            for i, group in enumerate(groups):
                print(f"  📦 Group {i+1}: {len(group['content'])} chars")
        
    except Exception as e:
        print(f"❌ Semantic grouper error: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    print("🚀 Starting debug tests...\n")
    
    if test_imports():
        if test_pdf_processing():
            if test_llm_client():
                if test_semantic_grouper():
                    print("\n🎉 All tests passed! The issue might be in the main processing loop.")
                else:
                    print("\n❌ Problem found in semantic grouper")
            else:
                print("\n❌ Problem found in LLM client")
        else:
            print("\n❌ Problem found in PDF processing")
    else:
        print("\n❌ Problem found in imports") 