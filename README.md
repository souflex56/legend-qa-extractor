# 🔍 Legend QA Extractor

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/souflex56/legend-qa-extractor)

**基于本地大模型Ollama的PDF问答对提取工具**<br>
**适用场景**：对话系统训练、知识库构建、智能客服数据准备、教育内容提取

</div>

## 📚 目录

- [🚀 技术亮点](#-技术亮点)
  - [QA-First处理架构](#qa-first处理架构)
  - [🎯 三层智能分块](#-三层智能分块)
  - [🔗 Prompt智能处理](#-prompt智能处理)
  - [⚡ 性能优化特性](#-性能优化特性)
- [🚀 快速开始](#-快速开始)
- [🎯 使用示例](#-使用示例)
- [📊 评估系统详解](#-评估系统详解)
- [🤝 贡献指南](#-贡献指南)
- [📄 开源协议](#-开源协议)

## 🚀 技术亮点

### QA-First处理架构

- 全文扫描识别：在全文范围内识别完整问答对（可配置questioner_prefixes），不受传统分块限制
- 规则直接提取：高置信度问答对直接提取，无需LLM处理，效率提升
- 智能配对逻辑：支持多种问答组合模式（如问-答、问-问-答、问-答-答）

### 🎯 三层智能分块

#### L1 规则预筛选（高置信度快速识别）

- 标准格式识别：支持questioner_prefixes和answerer_prefixes定义的问答标识
- 支持扩展模式：如编号前缀（1.网友:）、带标识符（网友A:）、间接引用（有人问:）
- 高效直接处理：直接提取，无需语义分析+LLM 处理，提高效率

```python
# 支持的问答格式示例
"网友：什么是价值投资？"  # 基础格式
"1. 主持人：如何看待市场波动？"  # 编号前缀
"有人问：stop doing list是什么？"  # 间接引用
```

#### L2 语义动态分组（智能边界处理）

- 动态阈值计算：基于文本语义自动调整相似度阈值
- 潜在问题检测：结合长度限制、疑问词、词性分析
- 领域自适应：根据文本内容选择通用或领域特定模型（如paraphrase-multilingual-MiniLM-L12-v2）
- 提供关键参数：max_question_length, default_similarity_threshold, std_factor

#### L3 块优化与合并（大小规范化）

- 智能合并：合并过小的语义相关块（min_block_size）
- 自动分割：按语义分割超大块（max_block_size）
- 置信度维持：当系统合并多个文本块时，每个块都有自己的置信度等级（high/medium/low）。合并时保持最低置信度，保持处理的保守性和可靠性

### 🔗 Prompt智能处理

#### 智能Prompt长度管理

- 动态模板选择：根据Token数量自动切换Prompt模板
  - Full Prompt（`full_prompt`）：Token充足时使用，包含详细示例
  - Compact Prompt（`compact_prompt`）：Token受限时自动切换，保留核心指令
- 模板目录配置：`prompt_template_dir`
- 自动截断策略：优先级按段落→句子→字符顺序

#### 长答案处理策略  

- 链式摘要机制：超长答案自动分片处理（chain_summary_threshold）
- NLI蕴含校验：使用mDeBERTa-v3模型验证摘要质量（nli_model_path）
- 预定义降级策略：生成式摘要 → 抽取式摘要 → 原文保留



```python
# 长答案处理流程
if len(answer) > 3000:  # 自定义长答案触发参数
    chunks = smart_split(answer, target_length=500)
    summaries = [llm_summarize(chunk, length=50) for chunk in chunks] 
    final_summary = " ".join(summaries)
    if nli_entailment_score(answer, final_summary) > 0.7:
        return final_summary
    else:
        return extractive_summary(answer)
```

#### Token监控与优化

- 实时Token估算：中文1.5倍，英文0.75倍
- 自动阈值管理：Token上限6000（max_prompt_tokens）
- 使用统计追踪：记录Full/Compact配置使用

### ⚡ 性能优化特性

- 🤖 本地大模型集成：支持Ollama自定义LLM model_name（默认：qwen2.5:7b-instruct）
- 批量并行处理：启用多线程并行，配置batch_size和max_workers
- 连接池优化：复用HTTP连接
- 模型预热机制：配置模型预热，保持30分钟热状态

## 🚀 快速开始

### 环境要求

- **Python**: 3.8 或更高版本
- **Ollama**: 已安装并运行 ([安装指南](https://ollama.ai/))
- **模型**: 推荐 `qwen2.5:7b-instruct` 或更高版本
- **新增依赖**: sentence-transformers, jieba, torch

### 一键安装

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/legend-qa-extractor.git
cd legend-qa-extractor

# 2. 自动环境设置（推荐）
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# 3. 拉取推荐模型（可自定义）
ollama pull qwen2.5:7b-instruct
```

### 配置示例

```yaml
# config/config.yaml - v2.0 配置
# 基础设置
pdf_filename: "document.pdf"
model_name: "qwen2.5:7b-instruct"
max_block_size: 1500
min_block_size: 200

# 🚀 智能语义分组配置
semantic_grouping:
  max_question_length: 50                    # 潜在问题最大长度
  default_similarity_threshold: 0.65         # 默认相似度阈值
  std_factor: 0.5                           # 动态阈值计算系数
  high_confidence_min_size: 100             # 高置信度块最小尺寸
  high_confidence_max_size: 2000            # 高置信度块最大尺寸
  model_name: "paraphrase-multilingual-MiniLM-L12-v2"

# 🚀 长答案处理配置
long_answer_processing:
  chain_summary_threshold: 3000             # 触发链式摘要的长度
  summary_length: 50                        # 摘要片段目标长度
  entailment_threshold: 0.7                 # NLI蕴含验证阈值
  nli_model_path: "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# 🚀 过滤和质量控制
filtering_level: "balanced"                 # strict/balanced/none
semantic_threshold: 0.5                     # 语义过滤阈值
```

## 🎯 使用示例

### 基础提取

```bash
# 高效提取
python extract_qa.py interview.pdf

# 高质量模式
python extract_qa.py document.pdf \
  --model qwen2.5:14b-instruct \
  --temperature 0.05 \
  --config config/high_quality.yaml

# 快速测试（按比例采样）
python extract_qa.py large_document.pdf --sample 0.02
```

### 评估工作流

```python
from evaluation import QAEvaluator

# 初始化评估器
evaluator = QAEvaluator(
    golden_set_path="golden_set.jsonl",
    generated_qa_path="output/extracted_qa.jsonl"
)

# 执行评估
results = evaluator.evaluate_qa_extraction()

# 生成报告
evaluator.generate_evaluation_report(results)
```

## 📊 评估系统详解

### 评估指标体系

#### 语义相似度评估
- **问题匹配度**：使用多语言sentence-transformer计算问题相似度
- **答案质量度**：深度语义理解，不仅比较文字，更关注语义
- **综合评分**：问题相似度(30%) + 答案相似度(70%)

#### 详细质量报告
```json
{
  "golden_set_size": 7,
  "generated_set_size": 7,
  "average_question_similarity": 0.9983,
  "average_answer_similarity": 0.9279,
  "overall_score": 0.9490,
  "matched_pairs": 7,
  "grade": "优秀 🏆",
  "confidence_distribution": {
    "high": "7 blocks (100.0%)"
  }
}
```

### 快速开始评估

```bash
# 1. 从Excel创建黄金标准集
python scripts/excel_to_golden_set.py template  # 生成标注集模板
# 编辑 golden_set_template.xlsx
python scripts/excel_to_golden_set.py convert   # 转换为JSONL

# 2. 运行提取
python extract_qa.py your_document.pdf

# 3. 执行评估
python evaluation.py
```

### Token监控

实时监控和优化token使用：

```bash
📊 Token使用报告
==================================================
📝 Prompt使用统计:
   精简版: 15 次 (60%)
   完整版: 10 次 (40%)
🎯 Token使用统计:
   平均利用率: 68.5%
   最高使用: 3,245 tokens
   最低使用: 1,892 tokens
⚡ 性能指标:
   处理速度: 2.3 块/分钟
   平均响应时间: 8.7秒
🟢 Token利用率健康
```

### Excel工作流

#### 1. 创建标准集模板
```bash
python scripts/excel_to_golden_set.py template
```

#### 2. Excel表格编辑

| question         | answer                     | domain     | difficulty | quality_score |
|------------------|----------------------------|------------|------------|---------------|
| 什么是价值投资？ | 价值投资就是买股票就是公司... | investment | medium     | 5             |
| 如何看待市场波动？| 市场先生的报价每天都不一样... | investment | easy       | 4             |

#### 3. 转换并评估
```bash
python scripts/excel_to_golden_set.py convert
python evaluation.py
```

## 🤝 贡献指南

我们欢迎以下方向的贡献：

- 🧠 新的领域识别模型
- 📝 更好的摘要算法  
- 🌍 多语言支持（目前主要支持中文）
- ⚡ 性能优化建议
- 🐛 Bug修复和代码优化
- 📖 文档完善和示例补充

## 📄 开源协议

MIT License




