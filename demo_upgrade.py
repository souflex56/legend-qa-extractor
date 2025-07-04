#!/usr/bin/env python3
"""
智能分块升级演示脚本
演示 legend-qa-extractor 的新功能：智能分块、主题锚点、滑动上下文

运行方式：
python demo_upgrade.py
"""

import os
import sys
import logging
from src.config import load_config
from src.core.smart_block_processor import SmartBlockProcessor
from src.core import TextProcessor, LLMClient

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_smart_blocking():
    """演示智能分块功能"""
    
    print("🚀 legend-qa-extractor 智能分块升级演示")
    print("="*60)
    
    # 加载配置
    config = load_config('config/config.yaml')
    logger.info("配置加载成功")
    
    # 初始化依赖组件
    text_processor = TextProcessor(known_prefixes=config.known_prefixes)
    
    try:
        llm_client = LLMClient(
            host=config.ollama_host,
            model_name=config.model_name
        )
        logger.info("LLM客户端初始化成功")
    except Exception as e:
        logger.error(f"LLM客户端初始化失败: {e}")
        print("❌ 请确保 Ollama 服务正在运行")
        return
    
    # 初始化智能分块处理器
    smart_processor = SmartBlockProcessor(
        text_processor=text_processor,
        llm_client=llm_client,
        max_block_size=config.max_block_size,
        min_block_size=config.min_block_size,
        qa_allowance_ratio=config.qa_allowance_ratio,
        enable_sliding_context=config.enable_sliding_context,
        enable_llm_anchor=config.enable_llm_anchor,
        anchor_keywords_count=config.anchor_keywords_count
    )
    
    print(f"✅ SmartBlockProcessor 初始化成功")
    print(f"   📊 配置: max_size={config.max_block_size}, min_size={config.min_block_size}")
    print(f"   🔗 滑动上下文: {'启用' if config.enable_sliding_context else '禁用'}")
    print(f"   🏷️  LLM锚点: {'启用' if config.enable_llm_anchor else '禁用'}")
    
    # 示例文本
    sample_text = """
一、价值投资的核心理念

网友：段总，您能谈谈价值投资的核心是什么吗？

段永平：价值投资的核心就是买股票就是买公司，买公司就是买这家公司的生意。很多人炒股票，实际上把股票当作筹码在炒，这是不对的。

二、关于护城河

文章引用：巴菲特说过，护城河是企业最重要的竞争优势。

段：护城河确实很重要。好的护城河就是别人没法抢你的生意，或者抢起来很困难。比如可口可乐的品牌，这就是护城河。

网友：那怎么判断一个公司有没有护城河呢？

段：最简单的方法就是看这个公司能不能随便涨价，如果可以随便涨价而客户还是要买，那就可能有护城河。

三、关于市场波动

有人认为市场波动是投资者最大的敌人。

段永平：市场先生的报价每天都不一样，你要把他当成一个躁郁症患者，不要太把他的话当真。波动其实是我们的朋友，不是敌人。
"""
    
    print(f"\n📝 示例文本长度: {len(sample_text)} 字符")
    print("\n🔄 开始智能分块处理...")
    
    # 执行智能分块
    try:
        blocks = smart_processor.process_document_into_blocks(sample_text)
        
        print(f"\n✅ 智能分块完成!")
        print(f"   📦 生成块数: {len(blocks)}")
        
        # 展示结果
        for i, block in enumerate(blocks, 1):
            print(f"\n{'='*50}")
            print(f"🔹 块 {i} (长度: {len(block['content'])} 字符)")
            
            if 'anchor' in block:
                print(f"🏷️  主题锚点: {block['anchor']}")
            
            if 'sliding_context' in block:
                print(f"🔗 滑动上下文: {block['sliding_context'][:100]}...")
            
            print(f"📄 内容预览:")
            content_preview = block['content'][:200].replace('\n', ' ')
            print(f"   {content_preview}...")
        
        print(f"\n🎉 演示完成! 新的智能分块功能已成功运行")
        
    except Exception as e:
        logger.error(f"智能分块处理失败: {e}")
        print(f"❌ 处理失败: {e}")

def show_upgrade_summary():
    """显示升级总结"""
    print("\n" + "="*80)
    print("🎯 legend-qa-extractor 升级总结 (第一阶段完成)")
    print("="*80)
    
    print("✅ 已完成的改进:")
    print("   1. 📦 新增 SmartBlockProcessor - 智能文本分块核心引擎")
    print("   2. 🏗️  四层分块策略:")
    print("      - 第一层: 基于结构的硬分块")
    print("      - 第二层: 智能自适应合并")
    print("      - 第三层: 质量保障过滤")
    print("      - 第四层: 元数据生成 (锚点 + 上下文)")
    print("   3. 🏷️  LLM生成主题锚点 - 为每个文本块提取核心关键词")
    print("   4. 🔗 滑动上下文 - 跨块关联增强理解")
    print("   5. ⚙️  配置文件更新 - 支持所有新功能开关")
    print("   6. 📊 评估脚本 - 第二阶段量化评估体系")
    
    print("\n🎮 新增配置参数:")
    print("   - qa_allowance_ratio: QA内容块大小容忍度 (默认 1.2)")
    print("   - enable_sliding_context: 启用滑动上下文 (默认 true)")  
    print("   - enable_llm_anchor: 启用LLM锚点生成 (默认 true)")
    print("   - anchor_keywords_count: 锚点关键词数量 (默认 3)")
    
    print("\n📋 使用说明:")
    print("   1. 🔧 配置: 编辑 config/config.yaml 调整参数")
    print("   2. 🏃 运行: python extract_qa.py (使用新的智能分块)")
    print("   3. 📊 评估: python evaluation.py (需要先创建黄金标准集)")
    
    print("\n🔮 第二阶段预告:")
    print("   - 创建黄金标准集 (golden_set.jsonl)")
    print("   - 运行量化评估获得系统评分")
    print("   - 基于评分结果进行targeted优化")
    
    print("\n💡 立即体验:")
    print("   python demo_upgrade.py  # 运行此演示")
    print("   python extract_qa.py   # 使用新功能处理实际PDF")
    
    print("="*80)

def main():
    """主函数"""
    try:
        show_upgrade_summary()
        print("\n🚀 开始功能演示...")
        demo_smart_blocking()
        
    except KeyboardInterrupt:
        print("\n\n👋 演示中断")
    except Exception as e:
        logger.error(f"演示失败: {e}")
        print(f"❌ 演示失败: {e}")

if __name__ == "__main__":
    main() 