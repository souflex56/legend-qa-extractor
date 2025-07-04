# Token优化指南

## 问题描述

您的系统经常出现 `"truncating input prompt"` 警告，原因是输入prompt超过了模型的4096 token限制。这会导致：
- 重要信息被截断
- 提取质量下降
- 处理效率降低

## 解决方案概览

我们实施了一套全面的token管理优化方案：

### 1. 智能Prompt管理

#### 双版本Prompt系统
- **精简版**：约200 tokens，保留核心功能
- **完整版**：约1500 tokens，包含详细规则和示例
- 系统根据可用空间自动选择最适合的版本

#### 动态长度控制
```python
# 新增的token估算功能
def estimate_token_count(text: str) -> int:
    """估算文本的token数量（中文约1.5倍字符数）"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars * 0.75)
```

### 2. 优化配置参数

```yaml
# 新的推荐配置
max_block_size: 1000              # 从1500降至1000
min_block_size: 200               # 从300降至200  
qa_allowance_ratio: 1.1           # 从1.2降至1.1
max_prompt_tokens: 3800           # 新增：设置token上限
enable_token_monitoring: true     # 新增：自动token监控和报告
enable_sliding_context: false     # 暂时关闭以节省空间
enable_llm_anchor: false          # 暂时关闭以减少调用
```

### 3. 智能文本截断

#### 按优先级截断
1. **段落级截断**：优先保持段落完整性
2. **句子级截断**：其次按句子边界截断
3. **字符级截断**：最后保证不超过限制

```python
def _smart_truncate_text(self, text: str, max_chars: int) -> str:
    """智能截断文本，尽量保持完整性"""
    # 优先按段落截断
    paragraphs = text.split('\n\n')
    # 如果段落截断效果不佳，按句子截断
    # 最后确保不超长
```

## 使用指南

### 1. 运行Token分析

使用我们提供的监控工具分析当前配置：

```bash
# 分析默认配置
python scripts/monitor_token_usage.py

# 分析自定义配置
python scripts/monitor_token_usage.py --config your_config.yaml

# 运行示例测试
python scripts/monitor_token_usage.py --test
```

### 2. 查看分析报告

分析报告包含：
- 基础prompt token使用情况
- 不同场景下的token预估
- 个性化优化建议
- 推荐配置参数

示例输出：
```
🔍 Token使用分析报告
==================================================
📊 基础Prompt分析:
   精简版prompt: 245 tokens (326 字符)
   完整版prompt: 1847 tokens (2456 字符)

⚙️  当前配置:
   最大块大小: 1000 字符
   Token限制: 3800 tokens
   
📈 Token使用预估:
   场景1 (最大块+精简prompt): 1745 tokens
   场景2 (最大块+完整prompt+上下文): 3347 tokens
   
💡 优化建议:
   ✅ 精简模式下块大小合适
   ✅ 完整prompt模式下配置合适
   
🎯 推荐配置:
   推荐最大块大小: 2333 字符
   当前token利用率: 45.9%
   🟢 利用率合理
```

### 3. 自动Token监控（新功能）

系统现在会在每次QA提取任务完成后自动输出详细的token使用总结：

```yaml
# 配置文件中启用自动监控
enable_token_monitoring: true  # 默认启用
```

**自动总结报告示例：**
```
📊 Token使用总结报告
==================================================
🔢 处理块数: 25
📝 Prompt使用统计:
   精简版: 20 次
   完整版: 5 次
🎯 Token使用统计:
   平均使用: 1650 tokens
   最大使用: 2200 tokens
   最小使用: 1200 tokens
   平均利用率: 43.4%
🟢 Token利用率健康
✅ 无文本截断发生
==================================================
```

### 4. 运行时监控状态

在处理过程中关注以下日志：

```
# 正常情况 - 使用完整prompt
DEBUG - Using full prompt, estimated tokens: 2847
DEBUG - Block token usage: 2847/3800 tokens (75.0%)

# Token不足 - 自动降级为精简prompt  
DEBUG - Using compact prompt, estimated tokens: 1245
DEBUG - Block token usage: 1245/3800 tokens (32.8%)

# 需要截断 - 文本过长
WARNING - Text block truncated to 800 chars due to token limit
WARNING - ⚠️ Potential truncation detected: 4200 tokens > 3800 limit

# 自动总结（处理完成后）
INFO - 📊 Token使用总结报告
```

## 性能优化建议

### 1. 根据文档类型调整配置

#### 对话类文档（QA密集）
```yaml
max_block_size: 800
enable_qa_filter: true
extract_ratio: 1.0
```

#### 长文类文档（内容丰富）
```yaml
max_block_size: 1200
enable_qa_filter: false
extract_ratio: 0.5
```

### 2. 渐进式优化流程

1. **第一轮**：使用保守配置，确保不会截断
2. **第二轮**：根据实际token使用情况微调
3. **第三轮**：启用高级功能（sliding_context, llm_anchor）

### 3. 批量处理策略

对于大文档，建议：
- 使用 `extract_ratio` 先处理一部分
- 监控token使用情况
- 调整参数后处理剩余部分

## 故障排除

### 常见问题

#### 1. 仍然出现截断警告
**原因**：文档包含特别长的段落
**解决**：降低 `max_block_size` 或启用更积极的文本截断

#### 2. 提取质量下降
**原因**：过度使用精简prompt
**解决**：增加 `max_prompt_tokens` 限制或减小 `max_block_size`

#### 3. 处理速度慢
**原因**：文本块过小，数量过多
**解决**：适当增加 `min_block_size`

### 调试技巧

1. **启用详细日志**：
```yaml
log_level: DEBUG
```

2. **使用监控脚本**：
```bash
python scripts/monitor_token_usage.py --test
```

3. **检查具体prompt**：
在日志中搜索 "estimated tokens" 来查看实际使用情况

## 最佳实践

1. **配置管理**：
   - 为不同类型的文档创建专门的配置文件
   - 定期运行token分析以优化参数

2. **监控和调整**：
   - 处理前先运行token分析
   - 根据实际使用情况调整配置
   - 关注错误日志中的截断警告

3. **性能平衡**：
   - 在提取质量和处理速度间找到平衡
   - 优先保证不出现截断，再优化其他指标

## 版本历史

- **v1.0**：初始token管理实现
- **v1.1**：添加智能截断和双版本prompt
- **v1.2**：新增token监控工具和分析报告

---

通过这些优化措施，您的系统应该能够有效避免prompt截断问题，同时保持良好的提取质量和处理效率。 