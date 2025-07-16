# SemanticGrouper 合并说明

## 概述

本项目已将 `SemanticProcessor` 的功能合并到 `SemanticGrouper` 中，提供统一的语义分组接口。合并后的 `SemanticGrouper` 具有以下特点：

- **功能完整**：整合了两个文件的所有功能
- **接口兼容**：保持原有接口的同时，提供新的兼容接口
- **配置灵活**：支持多种配置选项和过滤级别
- **性能优化**：统一的模型加载和缓存机制

## 主要功能

### 1. 原有功能（SemanticGrouper）

- **三层分块策略**：
  - 规则预筛选（高置信度识别）
  - 语义动态分组（复杂边界处理）
  - 块优化和合并

- **高置信度分组**：
  - 严格的一问一答配对
  - 特殊size限制策略
  - 领域模型支持

### 2. 新增功能（来自SemanticProcessor）

- **兼容接口**：`group_text_by_semantics()` 方法
- **过滤级别**：strict、balanced、none
- **保守规则**：更安全的问答配对策略
- **简化语义分组**：使用sklearn的cosine_similarity

## 使用方法

### 基本使用

```python
from src.core.semantic_grouper import SemanticGrouper

# 配置
config = {
    'max_block_size': 1500,
    'min_block_size': 200,
    'filtering_level': 'balanced',
    'semantic_threshold': 0.5,
    'semantic_grouping': {
        'high_confidence_min_size': 100,
        'high_confidence_max_size': 2000,
        'model_name': 'paraphrase-multilingual-MiniLM-L12-v2'
    }
}

# 初始化
grouper = SemanticGrouper(config)

# 方法1：处理段落列表（原有接口）
paragraphs = ["网友：什么是价值投资？", "段永平：价值投资就是买股票就是买公司。"]
blocks = grouper.group(paragraphs)

# 方法2：处理原始文本（新接口）
text = "网友：什么是价值投资？\n\n段永平：价值投资就是买股票就是买公司。"
groups = grouper.group_text_by_semantics(text)
```

### 过滤级别

```python
# 严格模式：只返回高置信度组
config_strict = {'filtering_level': 'strict'}
groups = grouper.group_text_by_semantics(text, config_strict)

# 平衡模式：返回高置信度和中置信度组
config_balanced = {'filtering_level': 'balanced'}
groups = grouper.group_text_by_semantics(text, config_balanced)

# 无过滤：返回所有组
config_none = {'filtering_level': 'none'}
groups = grouper.group_text_by_semantics(text, config_none)
```

## 配置选项

### 语义分组配置

```yaml
semantic_grouping:
  # 最大问题长度
  max_question_length: 50
  
  # 默认相似度阈值
  default_similarity_threshold: 0.65
  
  # 标准差系数
  std_factor: 0.5
  
  # 高置信度块的最小size
  high_confidence_min_size: 100
  
  # 高置信度块的最大size
  high_confidence_max_size: 2000
  
  # 模型名称
  model_name: "paraphrase-multilingual-MiniLM-L12-v2"
  
  # 设备配置
  device: null
```

### 过滤配置

```yaml
# 过滤级别
filtering_level: "balanced"  # strict, balanced, none

# 语义相似度阈值
semantic_threshold: 0.5
```

## 输出格式

### group() 方法输出

```python
[
    {
        'content': '问答内容',
        'confidence': 'high|medium|low',
        'type': 'rule_based_strict|semantic|semantic_merge',
        'indices': [0, 1],  # 原始段落索引
        'similarity_score': 0.85,  # 语义相似度分数（可选）
        'domain': 'general'  # 检测的领域（可选）
    }
]
```

### group_text_by_semantics() 方法输出

```python
[
    {
        'content': '问答内容',
        'type': 'high-confidence|medium-confidence|low-confidence'
    }
]
```

## 迁移指南

### 从 SemanticProcessor 迁移

如果你之前使用 `SemanticProcessor`，只需要：

1. **导入更改**：
   ```python
   # 之前
   from src.core.semantic_processor import SemanticProcessor
   
   # 现在
   from src.core.semantic_grouper import SemanticGrouper
   ```

2. **初始化更改**：
   ```python
   # 之前
   processor = SemanticProcessor(text_processor, model_name)
   
   # 现在
   grouper = SemanticGrouper(config)
   ```

3. **方法调用保持不变**：
   ```python
   # 两个版本都支持
   groups = grouper.group_text_by_semantics(text, config)
   ```

### 从旧版 SemanticGrouper 迁移

如果你之前使用 `SemanticGrouper`，无需更改代码，所有原有功能都保持不变。

## 性能优化

### 模型缓存

- 模型只加载一次，在实例化时缓存
- 支持GPU加速（通过device配置）
- 自动错误处理和重试机制

### 内存优化

- 大块自动分割
- 小块智能合并
- 索引追踪优化

## 测试

运行测试验证合并功能：

```bash
python test_merged_semantic_grouper.py
```

测试包括：
- 原有group方法
- 新增group_text_by_semantics方法
- 不同过滤级别
- block_has_qa方法
- 保守规则分组

## 注意事项

1. **依赖要求**：需要安装 `sentence-transformers` 和 `sklearn`
2. **模型下载**：首次使用会自动下载模型文件
3. **内存使用**：大文档处理时注意内存使用情况
4. **配置优先级**：配置文件中的值会覆盖代码默认值

## 未来扩展

- 支持更多领域模型
- 增加更多语义分组算法
- 优化大文档处理性能
- 增加更多配置选项 