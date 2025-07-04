#!/usr/bin/env python3
"""
Token使用监控脚本

该脚本帮助监控和分析QA提取过程中的token使用情况，
提供优化建议以避免prompt截断问题。
"""

import sys
import os
import argparse
import re
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, Config
from src.core.qa_extractor import QAExtractor
from src.core.text_processor import TextProcessor
from src.core.smart_block_processor import SmartBlockProcessor
from src.core.llm_client import LLMClient
from src.utils import setup_logger


def analyze_token_usage(config_path: str = None):
    """分析当前配置下的token使用情况"""
    
    # 加载配置
    if config_path:
        config = load_config(config_path)
    else:
        config = Config()
    
    print("🔍 Token使用分析报告")
    print("=" * 50)
    
    # 初始化组件
    qa_extractor = QAExtractor(max_prompt_tokens=config.max_prompt_tokens)
    text_processor = TextProcessor(known_prefixes=config.known_prefixes)
    
    # 分析基础prompt长度
    compact_tokens = qa_extractor.estimate_token_count(qa_extractor.compact_prompt)
    full_tokens = qa_extractor.estimate_token_count(qa_extractor.full_prompt)
    
    print(f"📊 基础Prompt分析:")
    print(f"   精简版prompt: {compact_tokens} tokens ({len(qa_extractor.compact_prompt)} 字符)")
    print(f"   完整版prompt: {full_tokens} tokens ({len(qa_extractor.full_prompt)} 字符)")
    
    # 分析当前配置
    print(f"\n⚙️  当前配置:")
    print(f"   最大块大小: {config.max_block_size} 字符")
    print(f"   最小块大小: {config.min_block_size} 字符")
    print(f"   Token限制: {config.max_prompt_tokens} tokens")
    print(f"   滑动上下文: {'启用' if config.enable_sliding_context else '禁用'}")
    print(f"   LLM锚点: {'启用' if config.enable_llm_anchor else '禁用'}")
    
    # 估算典型场景的token使用
    print(f"\n📈 Token使用预估:")
    
    # 场景1：最大块 + 精简prompt
    max_block_tokens = qa_extractor.estimate_token_count("测" * config.max_block_size)
    scenario1_tokens = compact_tokens + max_block_tokens
    print(f"   场景1 (最大块+精简prompt): {scenario1_tokens} tokens")
    
    # 场景2：最大块 + 完整prompt + 上下文
    context_tokens = 50 if config.enable_sliding_context else 0
    anchor_tokens = 20 if config.enable_llm_anchor else 0
    scenario2_tokens = full_tokens + max_block_tokens + context_tokens + anchor_tokens
    print(f"   场景2 (最大块+完整prompt+上下文): {scenario2_tokens} tokens")
    
    # 提供建议
    print(f"\n💡 优化建议:")
    
    if scenario1_tokens > config.max_prompt_tokens:
        recommended_size = int((config.max_prompt_tokens - compact_tokens) / 1.5)
        print(f"   ⚠️  当前最大块大小过大，建议降至 {recommended_size} 字符")
    else:
        print(f"   ✅ 精简模式下块大小合适")
    
    if scenario2_tokens > config.max_prompt_tokens:
        print(f"   ⚠️  完整prompt模式可能超限，建议:")
        print(f"      - 减小块大小至 {int((config.max_prompt_tokens - full_tokens) / 1.5)} 字符")
        print(f"      - 或禁用滑动上下文和LLM锚点功能")
    else:
        print(f"   ✅ 完整prompt模式下配置合适")
    
    # 计算理论最优配置
    print(f"\n🎯 推荐配置:")
    optimal_block_size = min(
        int((config.max_prompt_tokens - compact_tokens - 100) / 1.5),  # 为安全预留100tokens
        config.max_block_size
    )
    print(f"   推荐最大块大小: {optimal_block_size} 字符")
    
    utilization = scenario1_tokens / config.max_prompt_tokens * 100
    print(f"   当前token利用率: {utilization:.1f}%")
    
    if utilization > 90:
        print(f"   🔴 利用率过高，容易触发截断")
    elif utilization > 75:
        print(f"   🟡 利用率较高，建议监控")
    else:
        print(f"   🟢 利用率合理")


def test_sample_text():
    """使用示例文本测试token使用"""
    sample_text = """
网友：段总，什么是价值投资的核心？

段永平：价值投资的核心就是买股票就是买公司，买公司就是买这家公司的生意。很多人炒股票，实际上把股票当作筹码在炒，这是不对的。你要知道你买的是什么公司，这个公司做什么生意，这个生意是不是你看得懂的。

问：那怎么判断一个公司的生意是好是坏呢？

段：这个问题很好。一个简单的标准就是看这个公司能不能随便涨价，如果客户还是要买，那这个生意就可能有护城河。比如可口可乐，它涨价你还是要买，这就是护城河。

文章引用：巴菲特说过，时间是好公司的朋友，坏公司的敌人。

段：这话很有道理。好公司会越来越值钱，坏公司时间长了就会暴露问题。所以投资要买好公司，然后耐心等待。
    """
    
    config = Config()
    qa_extractor = QAExtractor(max_prompt_tokens=config.max_prompt_tokens)
    
    print(f"\n📝 示例文本测试:")
    print(f"   文本长度: {len(sample_text)} 字符")
    
    # 测试不同prompt组合
    prompts = [
        ("精简版", qa_extractor.compact_prompt + "\n\n" + sample_text),
        ("完整版", qa_extractor.full_prompt + "\n\n" + sample_text),
    ]
    
    for name, prompt in prompts:
        tokens = qa_extractor.estimate_token_count(prompt)
        status = "✅" if tokens <= config.max_prompt_tokens else "❌"
        print(f"   {name}: {tokens} tokens {status}")


def main():
    parser = argparse.ArgumentParser(description="Token使用监控和分析工具")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--test", action="store_true", help="运行示例文本测试")
    
    args = parser.parse_args()
    
    try:
        analyze_token_usage(args.config)
        
        if args.test:
            test_sample_text()
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 