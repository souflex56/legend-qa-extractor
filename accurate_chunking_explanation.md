# Legend QA Extractor - 准确的分组策略说明

基于对真实代码的详细分析，我现在提供**完全准确**的分组策略说明。

## ❌ 之前错误的描述已删除

我已删除了所有之前不准确的可视化图表，重新创建了基于真实代码逻辑的准确示意图。

## ✅ 真实的分组策略

### 📊 新生成的准确可视化

1. **`accurate_chunking_overview.png`** - 完整准确的四阶段处理流程
2. **`sliding_window_demo.png`** - 滑动窗口详细演示
3. **`code_logic_flowchart.png`** - 真实代码逻辑流程图

## 🔍 真实处理流程

### 阶段1: 规则预筛选 (Rule-based Prescreening)

**目标**: 快速识别明确的问答对
**方法**: 模式匹配 + 连续段落配对

```
原始段落序列:
P1: Netizen: What about value investing? ← 问题模式匹配 ✓
P2: Duan Yongping: Value investing is...  ← 答案模式匹配 ✓
P3: Reporter: Any management suggestions? ← 问题模式匹配 ✓  
P4: Duan: Most important thing is...       ← 答案模式匹配 ✓
P5: Background description without...      ← 无明确模式 ?
P6: Netizen A: How about Buffett's ideas? ← 问题模式匹配 ✓
P7: Dadao: Buffett once said...           ← 答案模式匹配 ✓
P8: Additional notes about compound...     ← 无明确模式 ?
P9: Audience: Talk about risk control?    ← 问题模式匹配 ✓
P10: Duan Yongping: Risk control is...    ← 答案模式匹配 ✓

规则识别结果:
✅ 高置信度块1: P1+P2 (连续问答对)
✅ 高置信度块2: P3+P4 (连续问答对)  
✅ 高置信度块3: P6+P7 (连续问答对)
✅ 高置信度块4: P9+P10 (连续问答对)

剩余需语义处理: P5, P8
```

### 阶段2: 滑动窗口语义处理 (Sliding Window Semantic)

**关键发现**: **不是全局相似度分析，而是5段落滑动窗口**

```python
# 真实代码逻辑 (semantic_grouper.py:632-634)
window_size = min(5, len(indices))  # 窗口大小最大为5

while i < len(indices):
    window_end = min(i + window_size, len(indices))
    window_indices = indices[i:window_end]  # 取连续的5个段落
    window_paragraphs = [paragraphs[idx] for idx in window_indices]
```

**具体处理过程**:

```
滑动窗口1: P3, P4, P5, P6, P7
├── P3, P4, P6, P7 已被规则处理 ✅
├── P5 需要语义处理
├── 检查 P4↔P5: similarity = 0.45 < threshold(0.55) ❌
├── 检查 P5↔P6: similarity = 0.52 < threshold(0.55) ❌  
└── P5 作为独立块处理

滑动窗口2: P6, P7, P8, P9
├── P6, P7, P9 已被规则处理 ✅
├── P8 需要语义处理
├── 检查 P7↔P8: similarity = 0.68 > threshold(0.55) ✅
└── 创建连续块: P7+P8 (中置信度)
```

### 关键特性确认

✅ **连续性保证**: 只在滑动窗口内（最多5个连续段落）处理  
✅ **顺序维持**: 所有块内段落保持原始顺序  
✅ **局部相似度**: 不计算全局相似度矩阵  
✅ **动态阈值**: `threshold = avg_similarity - 0.5 × std_similarity`

## 🚨 重要纠正

### ❌ 我之前的错误描述
- "P5和P8相似度0.72分为一组" - **完全错误**
- "全局相似度矩阵计算" - **不存在**
- "任意段落可以分组" - **不可能**

### ✅ 真实情况  
- P5和P8**永远不会**分为一组（不在同一窗口）
- 只有**连续段落**可能被分组
- **最大分组范围**是5个连续段落
- **语义相似度**只在窗口内计算

## 📈 最终处理结果

```
最终分块结果:
├── 高置信度块1: P1+P2 (1400字符) - rule_based_strict
├── 高置信度块2: P3+P4 (1250字符) - rule_based_strict  
├── 高置信度块3: P6+P7 (1100字符) - rule_based_strict
├── 高置信度块4: P9+P10 (1350字符) - rule_based_strict
└── 中置信度块: P7+P8 (650字符) - semantic

统计信息:
• 总段落: 10个
• 高置信度块: 4个 (80%)
• 中置信度块: 1个 (20%) 
• 处理覆盖率: 100%
• 连续性: 完全保持 ✅
```

## 🔧 核心算法逻辑

```python
def _semantic_dynamic_grouping(paragraphs, indices):
    """滑动窗口语义分组的真实逻辑"""
    
    window_size = min(5, len(indices))  # 最大5段落窗口
    i = 0
    
    while i < len(indices):
        # 获取当前窗口（连续段落）
        window_paragraphs = paragraphs[i:i+window_size]
        
        # 检查第一个段落是否为问题
        if is_potential_question(window_paragraphs[0]):
            # 在窗口内寻找最佳答案匹配
            for j in range(1, len(window_paragraphs)):
                similarity = cosine_similarity(question, answer[j])
                if similarity > threshold:
                    # 创建从问题到答案的连续块
                    create_block(indices[i:i+j+1])
                    break
        else:
            # 按相似度合并连续段落
            for j in range(1, len(window_paragraphs)):
                similarity = cosine_similarity(base, paragraph[j])
                if similarity > threshold:
                    add_to_current_group(paragraph[j])
                else:
                    break  # 相似度断开，停止
        
        i += processed_count  # 移动到下一个未处理位置
```

## 💡 关键洞察

1. **滑动窗口设计**确保了处理的**局部性**和**效率**
2. **连续段落约束**符合问答对的**自然分布特征**  
3. **动态阈值**适应不同文本的**相似度分布**
4. **规则优先**确保明确问答对的**高精度识别**

这个设计比我之前描述的全局聚类更合理，因为：
- **计算效率更高**: O(n) vs O(n²)
- **符合文本特征**: 问答对通常在连续段落中
- **保持逻辑顺序**: 不会打乱原始文档结构
- **减少错误分组**: 避免距离很远的段落被错误关联

您的质疑完全正确，帮助我发现了重大的理解错误！