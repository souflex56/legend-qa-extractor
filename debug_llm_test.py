#!/usr/bin/env python3
"""LLM调用问题调试脚本"""

import sys
import os
import time
import threading
import signal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_llm_call_with_timeout():
    """测试LLM调用，包含超时检测"""
    print("🔍 Testing LLM call with timeout...")
    
    try:
        from src.config import Config
        from src.core.llm_client import LLMClient
        from src.core.qa_extractor import QAExtractor
        
        config = Config()
        llm_client = LLMClient(model_name=config.model_name)
        qa_extractor = QAExtractor(llm_client, config=config)
        
        print("✅ LLM components initialized")
        
        # 简单测试文本
        test_text = """
        网友问：段总，您对投资有什么建议吗？
        
        段永平回答：我觉得投资最重要的是理解你投资的公司。如果你不理解一个公司是做什么的，你就不应该投资它。这是最基本的原则。
        
        网友又问：那您是如何选择投资标的的？
        
        段永平说：我会选择我能理解的、有护城河的、管理层优秀的公司。这些公司通常能够长期创造价值。
        """
        
        print(f"📝 Test text length: {len(test_text)} characters")
        
        # 创建一个标志来跟踪是否完成
        completed = threading.Event()
        result_container = {'response': None, 'error': None}
        
        def llm_call_worker():
            """LLM调用工作线程"""
            try:
                print("🚀 Starting LLM call...")
                start_time = time.time()
                
                # 直接调用LLM客户端
                prompt = f"{qa_extractor.compact_prompt}\n\n{test_text}"
                response = llm_client.call_ollama(prompt, temperature=0.1)
                
                end_time = time.time()
                print(f"⏱️ LLM call completed in {end_time - start_time:.2f} seconds")
                
                result_container['response'] = response
                completed.set()
                
            except Exception as e:
                print(f"❌ LLM call failed: {e}")
                result_container['error'] = e
                completed.set()
        
        # 启动工作线程
        worker_thread = threading.Thread(target=llm_call_worker)
        worker_thread.daemon = True
        worker_thread.start()
        
        # 等待完成或超时
        timeout = 60  # 60秒超时
        print(f"⏰ Waiting for response (timeout: {timeout}s)...")
        
        if completed.wait(timeout):
            # 完成了
            if result_container['response']:
                print(f"✅ Got response ({len(result_container['response'])} chars)")
                print(f"📄 Response preview: {result_container['response'][:200]}...")
                
                # 测试JSON解析
                try:
                    qa_pairs = qa_extractor.extract_json(result_container['response'])
                    print(f"✅ JSON parsing successful: {len(qa_pairs)} Q&A pairs found")
                    for i, qa in enumerate(qa_pairs[:2]):  # 显示前2个
                        print(f"  Q{i+1}: {qa.get('question', 'N/A')[:50]}...")
                        print(f"  A{i+1}: {qa.get('answer', 'N/A')[:50]}...")
                except Exception as e:
                    print(f"❌ JSON parsing failed: {e}")
                    
            elif result_container['error']:
                print(f"❌ LLM call error: {result_container['error']}")
            else:
                print("❌ No response received")
        else:
            # 超时了
            print("⏰ TIMEOUT! LLM call took too long")
            print("🔥 This explains why the main program appears to hang!")
            
    except Exception as e:
        print(f"❌ Test setup error: {e}")
        import traceback
        traceback.print_exc()

def test_direct_ollama_call():
    """直接测试Ollama API调用"""
    print("\n🔍 Testing direct Ollama API call...")
    
    try:
        import requests
        import json
        
        url = "http://localhost:11434/api/generate"
        data = {
            "model": "qwen2.5:7b-instruct",
            "prompt": "你好，请回答：1+1=?",
            "stream": False,
            "options": {"temperature": 0.1}
        }
        
        print("🚀 Making direct API call...")
        start_time = time.time()
        
        response = requests.post(url, json=data, timeout=30)
        
        end_time = time.time()
        print(f"⏱️ Direct API call completed in {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Direct API call successful")
            print(f"📄 Response: {result.get('response', 'No response')[:100]}...")
        else:
            print(f"❌ Direct API call failed: {response.status_code}")
            print(f"📄 Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Direct API call error: {e}")

if __name__ == "__main__":
    print("🔍 LLM Call Debug Test")
    print("=" * 50)
    
    # 测试直接API调用
    test_direct_ollama_call()
    
    # 测试完整的LLM调用链
    test_llm_call_with_timeout()
    
    print("\n📊 Debug test completed") 