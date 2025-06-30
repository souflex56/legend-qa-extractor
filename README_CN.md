# 🔍 Legend QA Extractor

<div align="right">

**Language** | **语言**: [🇺🇸 EN](README_EN.md) | [🇨🇳 中文](README.md)

</div>

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**使用本地大模型从PDF文档中提取问答对的专业工具**

*将您的PDF文档转换为结构化的问答数据集，用于AI训练*

</div>

## ✨ 核心特性

- 🤖 **本地大模型集成**: 使用 Ollama 配合 Qwen2.5 等模型，保护数据隐私
- 📄 **智能PDF处理**: 先进的文本提取和智能块分割技术
- 🎯 **智能问答识别**: 多模式识别各种问答格式
- ⚙️ **高度可配置**: 支持 YAML 配置文件和环境变量
- 🔧 **开发者友好**: 模块化架构，全面测试，命令行界面
- 📊 **质量指标**: 内置提取质量评估和详细日志
- 🚀 **生产就绪**: 类型提示，错误处理，专业代码结构

## 🔄 工作原理

```
📄 PDF输入 → 📝 文本分割 → 🔍 块过滤 → 🤖 AI分析 → 📊 问答对输出
               ↑ 块大小    ↑ 问答过滤    ↑ 模型 + 温度
            100-1500字符   开关 + 采样   qwen2.5:7b/自定义
                (可自定义)   0.1-1.0     温度: 0.0-1.0
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 已安装并运行 [Ollama](https://ollama.ai/)
- 需要处理的PDF文档

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/legend-qa-extractor.git
cd legend-qa-extractor

# 设置环境（创建虚拟环境，安装依赖）
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# 或手动设置
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 基本用法

```bash
# 从PDF提取问答对
python extract_qa.py document.pdf

# 指定输出文件
python extract_qa.py document.pdf -o my_output.jsonl

# 仅处理10%用于测试
python extract_qa.py document.pdf --sample 0.1

# 使用不同模型
python extract_qa.py document.pdf --model qwen2.5:14b-instruct

# 创建配置文件
python extract_qa.py --create-config

# 验证设置
python extract_qa.py --validate
```

## 📁 项目结构

```
legend-qa-extractor/
├── src/                          # 核心源代码
│   ├── config/                   # 配置管理
│   ├── core/                     # 核心处理模块
│   │   ├── pdf_processor.py      # PDF文本提取
│   │   ├── text_processor.py     # 文本分割和清理
│   │   ├── qa_extractor.py       # 问答对提取
│   │   └── llm_client.py         # Ollama集成
│   └── utils/                    # 工具函数
├── config/                       # 配置文件
├── examples/                     # 使用示例
├── tests/                        # 测试套件
├── scripts/                      # 设置和工具脚本
├── extract_qa.py                 # CLI入口点
└── output/                       # 生成结果
```

## ⚙️ 配置

### YAML配置文件

为您的特定需求创建配置文件：

```yaml
# config/config.yaml
pdf_filename: "your_document.pdf"
output_filename: "extracted_qa.jsonl"
model_name: "qwen2.5:7b-instruct"
max_block_size: 1500
extract_ratio: 1.0
enable_qa_filter: false
```

### 环境变量

使用环境变量覆盖设置：

```bash
export PDF_FILENAME="document.pdf"
export OLLAMA_MODEL="qwen2.5:14b-instruct"
export MAX_BLOCK_SIZE=2000
export EXTRACT_RATIO=0.5
```

### 命令行选项

```bash
python extract_qa.py --help

选项:
  --config CONFIG         YAML配置文件
  --output OUTPUT         输出JSONL文件
  --model MODEL          Ollama模型名称
  --sample RATIO         采样比例 (0.0-1.0)
  --enable-qa-filter     仅处理问答块
  --temperature TEMP     模型温度
  --log-level LEVEL      日志级别
  --validate             仅验证设置
```

## 🎯 高级用法

### 编程API

```python
from src.config import Config
from src.processor import QAExtractionProcessor

# 创建配置
config = Config()
config.pdf_filename = "document.pdf"
config.model_name = "qwen2.5:7b-instruct"
config.extract_ratio = 0.1  # 快速测试

# 初始化处理器
processor = QAExtractionProcessor(config)

# 验证设置
validation = processor.validate_setup()
if validation['valid']:
    # 处理PDF
    results = processor.process_pdf()
    print(f"提取了 {results['stats']['qa_pairs_extracted']} 个问答对")
```

### 自定义配置

```python
# 面试记录的自定义设置
config = Config()
config.known_prefixes.extend(["面试官", "候选人", "HR"])
config.max_block_size = 2000
config.enable_qa_filter = True
config.temperature = 0.05
```

## 📊 输出格式

工具生成结构化问答对的JSONL文件：

```json
{
  "question": "什么是价值投资？",
  "answer": "价值投资是一种投资策略，重点关注公司的内在价值...",
  "source_text": "网友：什么是价值投资？\n段永平：价值投资是一种..."
}
```

## 🔧 开发

### 设置开发环境

```bash
# 安装开发依赖
./scripts/setup_environment.sh --dev

# 安装pre-commit钩子
pre-commit install

# 运行测试
pytest

# 代码格式化
black src/ tests/
isort src/ tests/

# 类型检查
mypy src/
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行覆盖率测试
pytest --cov=src

# 运行特定测试
pytest tests/test_text_processor.py

# 运行集成测试
pytest -m integration
```

## 🏗️ 架构

### 核心组件

1. **PDFProcessor**: 从PDF文档提取文本
2. **TextProcessor**: 分割文本并应用清理
3. **QAExtractor**: 使用LLM识别和提取问答对
4. **LLMClient**: 管理与Ollama的通信
5. **Config**: 集中配置管理

## 🎛️ 配置选项

| 参数 | 描述 | 默认值 | 示例 |
|------|------|-------|------|
| `pdf_filename` | 要处理的PDF文件 | `"uploaded.pdf"` | `"interview.pdf"` |
| `model_name` | Ollama模型 | `"qwen2.5:7b-instruct"` | `"qwen2.5:14b"` |
| `max_block_size` | 最大文本块大小 | `1500` | `2000` |
| `extract_ratio` | 处理块的比例 | `1.0` | `0.1` |
| `enable_qa_filter` | 过滤问答模式块 | `false` | `true` |
| `temperature` | 模型创造性 | `0.1` | `0.05` |

## 📈 性能优化建议

1. **模型选择**: 使用 `qwen2.5:7b-instruct` 获得速度，使用 `qwen2.5:14b-instruct` 获得质量
2. **块大小**: 更大的块(2000+)提供更多上下文但处理更慢
3. **问答过滤**: 对有明确问答结构的文档启用以提高速度
4. **采样**: 使用 `extract_ratio=0.1` 进行快速测试
5. **批处理**: 使用不同配置处理多个文件

## 🛠️ 故障排除

### 常见问题

**Ollama连接失败**
```bash
# 检查Ollama是否运行
ollama serve

# 测试连接
curl http://localhost:11434/api/tags
```

**模型未找到**
```bash
# 拉取所需模型
ollama pull qwen2.5:7b-instruct
```

**提取质量低**
- 增加 `temperature` 获得更有创意的响应
- 调整 `max_block_size` 获得更好的上下文
- 启用 `enable_qa_filter` 进行专注处理

### 调试模式

```bash
# 启用详细日志
python extract_qa.py document.pdf --log-level DEBUG

# 检查验证
python extract_qa.py --validate
```

## 📚 示例

查看 `examples/` 目录了解：
- `run_example.py`: 编程使用示例
- `sample_config.yaml`: 配置模板
- 各种使用场景

## 🤝 贡献

我们欢迎贡献！请参阅我们的贡献指南：

1. Fork 仓库
2. 创建特性分支
3. 为新功能添加测试
4. 运行测试套件
5. 提交 pull request

### 开发设置

```bash
git clone https://github.com/yourusername/legend-qa-extractor.git
cd legend-qa-extractor
./scripts/setup_environment.sh --dev
pre-commit install
```

## 📄 许可证

此项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Ollama](https://ollama.ai/) 提供本地LLM基础设施
- [PyMuPDF](https://pymupdf.readthedocs.io/) 提供PDF处理
- 开源社区提供灵感和工具

## 📞 支持

- 📖 [文档](docs/)
- 🐛 [问题追踪](https://github.com/yourusername/legend-qa-extractor/issues)
- 💬 [讨论](https://github.com/yourusername/legend-qa-extractor/discussions)

---

<div align="center">

**如果觉得有帮助，请给个 ⭐ Star！**

</div> 