# Legend QA Extractor - 最终正确的分组策略可视化

## 📊 保留的正确文件

经过详细的代码分析和多次纠正，现在只保留**完全基于真实代码逻辑**的准确可视化文件：

### 🎯 核心可视化文件

1. **`remaining_paragraphs_grouping.png`** - 主要示意图
   - 展示P1、P2、P9、P10被规则过滤后的完整过程
   - 详细的滑动窗口语义处理演示
   - 包含两个5段落滑动窗口的具体处理逻辑

2. **`similarity_calculation_demo.png`** - 相似度计算详解
   - 窗口内相似度的具体计算过程
   - 动态阈值的计算公式和示例
   - SentenceTransformer的编码过程

3. **`remaining_paragraphs_grouping.py`** - 生成脚本
   - 基于真实代码逻辑的可视化生成脚本
   - 可重现所有图表

4. **`accurate_chunking_explanation.md`** - 准确的文字说明
   - 完全基于代码分析的正确描述
   - 纠正了之前所有的理解错误

## ✅ 已删除的错误文件

已删除所有之前不准确的文件：
- ❌ `chunking_strategy_visualization.py` (包含错误的全局相似度概念)
- ❌ `chunking_visualization_fixed.py` (仍有理解错误)  
- ❌ `accurate_chunking_visualization.py` (不够详细)
- ❌ `accurate_chunking_overview.png` (概念性演示，不够具体)
- ❌ `sliding_window_demo.png` (不够详细)
- ❌ `code_logic_flowchart.png` (抽象流程图)
- ❌ `chunking_strategy_analysis.md` (包含错误分析)
- ❌ `chunking_process_explanation.md` (包含错误描述)

## 🔍 核心纠正内容

### ❌ 之前的重大错误
1. **错误**: "P5和P8相似度0.72分为一组"
2. **错误**: "全局相似度矩阵计算"  
3. **错误**: "任意段落可以分组"
4. **错误**: "层次聚类处理"

### ✅ 真实情况
1. **正确**: P5和P8永远不会分组（不在同一滑动窗口）
2. **正确**: 只有窗口内局部相似度计算
3. **正确**: 只有连续段落可能分组
4. **正确**: 滑动窗口 + 连续段落处理

## 📈 真实的分组流程

```
输入: 10个段落 (P1-P10)

Step 1: 规则预筛选
├── P1+P2 → Block 1 (高置信度)
├── P3+P4 → Block 2 (高置信度)
├── P6+P7 → Block 3 (高置信度)
├── P9+P10 → Block 4 (高置信度)
└── 剩余: P5, P8

Step 2: 滑动窗口语义处理
├── 窗口1: [P3,P4,P5,P6,P7] → P5独立成块
└── 窗口2: [P6,P7,P8,P9,P10] → P7+P8分组

最终结果:
├── Block 1: P1+P2 (高置信度-规则)
├── Block 2: P3+P4 (高置信度-规则)  
├── Block 3: P6+P7 (高置信度-规则)
├── Block 4: P9+P10 (高置信度-规则)
├── Block 5: P5 (低置信度-语义)
└── Block 6: P7+P8 (中置信度-语义)
```

## 🎯 关键技术细节

### 滑动窗口参数
- **窗口大小**: `window_size = min(5, len(remaining_indices))`
- **处理方式**: 每次处理最多5个连续段落
- **移动策略**: `i += processed_count` (跳过已处理段落)

### 相似度计算
- **模型**: SentenceTransformer (`paraphrase-multilingual-MiniLM-L12-v2`)
- **方法**: `util.cos_sim(embedding1, embedding2).item()`
- **阈值**: `max(dynamic_threshold, default_0.55)`

### 动态阈值公式
```python
threshold = avg_similarity - std_factor × std_similarity
final_threshold = max(threshold, default_similarity_threshold)
```

## 💡 设计优势

1. **计算效率**: O(n) 滑动窗口 vs O(n²) 全局聚类
2. **保持逻辑**: 连续段落保持原文顺序
3. **符合直觉**: 问答对通常在连续文本中
4. **资源友好**: 局部计算降低内存需求

这套可视化完全基于真实代码逻辑，准确展示了Legend QA Extractor的**滑动窗口 + 连续段落**分组策略！