# Excel到Golden Set转换工具使用指南

这个工具可以帮你快速从Excel表格创建evaluation所需的`golden_set.jsonl`文件。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install pandas openpyxl
```

### 2. 创建Excel模板
```bash
python scripts/excel_to_golden_set.py template
```
这会生成 `golden_set_template.xlsx` 文件

### 3. 编辑Excel文件
用Excel/WPS/LibreOffice等打开 `golden_set_template.xlsx`，填入你的QA数据

### 4. 转换为JSONL
```bash
python scripts/excel_to_golden_set.py convert
```
这会生成 `golden_set.jsonl` 文件，可直接用于evaluation

## 📋 Excel表格格式

### 必需列（红色标记）
- **question**: 问题文本
- **answer**: 答案文本

### 可选列（绿色标记）
- **source_text**: 原始来源文本
- **domain**: 领域分类 (`general`/`investment`/`business`等)
- **difficulty**: 难度等级 (`easy`/`medium`/`hard`)
- **answer_type**: 答案类型 (`direct`/`explanatory`/`opinion`等)
- **quality_score**: 质量评分 (1-5分)
- **notes**: 备注说明

## 💡 使用技巧

### Excel编辑建议
1. **批量编辑**: 利用Excel的填充功能快速设置`domain`、`difficulty`等重复字段
2. **数据验证**: 对`quality_score`列设置1-5的数据验证
3. **颜色标记**: 用不同颜色标记不同难度或领域的题目
4. **冻结首行**: 冻结表头便于浏览大量数据

### 质量控制
1. **问题多样性**: 覆盖不同类型的问题（事实、观点、解释等）
2. **难度分布**: 建议easy:medium:hard = 3:5:2
3. **答案质量**: 确保答案准确、完整、简洁
4. **样本数量**: 建议30-50个高质量QA对

## 📊 示例数据

| question | answer | domain | difficulty | quality_score |
|----------|--------|--------|------------|---------------|
| 什么是stop doing list？ | 所谓要做对的事情实际上是通过不做不对的事情来实现的。 | business | medium | 5 |
| 价值投资的核心是什么？ | 价值投资的核心就是买股票就是买公司，买公司就是买这家公司的生意。 | investment | easy | 5 |

## 🔧 命令参考

```bash
# 创建Excel模板
python scripts/excel_to_golden_set.py template

# 执行转换
python scripts/excel_to_golden_set.py convert

# 自动模式（没有Excel文件时创建模板，有文件时转换）
python scripts/excel_to_golden_set.py
```

## ⚠️ 注意事项

1. **编码问题**: Excel文件会自动处理中文编码
2. **空值处理**: 空的可选字段会使用默认值
3. **数据验证**: 转换前会检查必需字段和数据格式
4. **文件路径**: 默认在当前目录生成文件，可修改脚本中的路径

## 🔄 工作流程

```
📝 编辑Excel → 🔄 运行转换 → 📊 生成JSONL → 🧪 执行评估
   ↑_______________________________________________|
                    优化QA对
```

## 🆘 常见问题

**Q: Excel文件格式错误怎么办？**
A: 工具会提示具体错误行和字段，根据提示修复即可

**Q: 可以只填必需字段吗？**
A: 可以，只要有`question`和`answer`列就能正常转换

**Q: 支持哪些Excel格式？**
A: 支持`.xlsx`格式，推荐使用Excel 2010+版本

**Q: 如何批量修改数据？**
A: 在Excel中使用查找替换、填充等功能，或直接编辑JSONL文件 

## 🛠️ 权限问题解决方案：

### 方式1：使用python命令运行（推荐）
```bash
python scripts/excel_to_golden_set.py convert
```

### 方式2：直接执行脚本（需要执行权限）
```bash
# 先添加执行权限（只需要做一次）
chmod +x scripts/excel_to_golden_set.py

# 然后可以直接运行
./scripts/excel_to_golden_set.py convert
# 或者
scripts/excel_to_golden_set.py convert
```

## 📊 看起来已经成功运行了！

从输出可以看到：
- ✅ 读取到 7 行数据（看来你已经添加了一些数据到Excel模板中）
- ✅ 成功转换了 7 个QA对
- ✅ 生成了 `golden_set.jsonl` 文件

## 💡 建议使用方式1

为了避免权限问题，建议使用 `python scripts/excel_to_golden_set.py` 的方式运行，这样不需要考虑文件权限。

## 🔄 完整工作流程：

```bash
<code_block_to_apply_changes_from>
```

现在你的Excel转换工具已经可以正常使用了！ 