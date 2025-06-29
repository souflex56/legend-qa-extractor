# 🚀 快速开始指南

## 1️⃣ 准备环境

### 安装Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

### 下载模型
```bash
ollama pull qwen2.5:7b-instruct
```

## 2️⃣ 启动应用

```bash
# 进入项目目录
cd gen_qa_ollama_pdf_streamlit

# 启动应用
./start_streamlit.sh
```

## 3️⃣ 使用步骤

1. **打开浏览器**: http://localhost:8501
2. **上传PDF**: 拖拽或选择PDF文件
3. **配置参数**: 在侧边栏调整设置
4. **开始处理**: 点击"🚀 开始生成QA对"
5. **查看结果**: 实时监控进度和预览结果
6. **下载数据**: 点击"📥 下载JSON格式"

## 🎯 推荐设置

### 快速测试
- 提取比例: 0.03 (3%)
- 最大块大小: 1500
- 最小块大小: 100
- QA过滤: 关闭

### 完整处理
- 提取比例: 1.0 (100%)
- 最大块大小: 2000
- 最小块大小: 200
- QA过滤: 开启

## ⚠️ 注意事项

- 确保Ollama服务正在运行
- 首次使用需要下载模型（约4GB）
- 大文件处理需要较长时间
- 建议在稳定网络环境下使用

## 🔧 故障排除

### Ollama连接失败
```bash
# 检查服务状态
curl http://localhost:11434/api/tags

# 重启服务
pkill ollama
ollama serve
```

### 模型未找到
```bash
# 查看已安装模型
ollama list

# 重新下载模型
ollama pull qwen2.5:7b-instruct
``` 