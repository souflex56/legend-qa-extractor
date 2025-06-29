#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, List
import json
import re
import fitz  # PyMuPDF
import logging
import random
import os
from tqdm import tqdm
from ollama import Client

# ==================== 配置区 ====================
# 获取脚本文件所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ 文件名配置 - 只需修改文件名即可
PDF_FILENAME = "uploaded.pdf"
OUTPUT_FILENAME = "output_final_duan-qa.jsonl"

# ✅ 自动拼接完整路径
PDF_PATH = os.path.join(SCRIPT_DIR, PDF_FILENAME)

# 创建输出文件夹
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 所有输出文件都放在output文件夹中
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
ERROR_LOG_PATH = os.path.join(OUTPUT_DIR, "extraction_errors_final.log")
SUCCESS_LOG_PATH = os.path.join(OUTPUT_DIR, "extraction_success_final.log")

# ✅ 模型配置 - 可通过环境变量控制
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

MAX_BLOCK_SIZE = 1500
MIN_BLOCK_SIZE = 100
# ✅ 文本提取比例，可自定义，0.03 表示顺序提取前 3% 的块，1 表示提取所有块
EXTRACT_RATIO = 1

# ✅ 是否开启问答检测过滤（True = 只处理含问答块；False = 保证完整性，所有块都处理）
ENABLE_QA_FILTER = False
# =================================================

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 错误日志
error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler(ERROR_LOG_PATH, mode='w', encoding='utf-8')
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - BLOCK_START\n%(message)s\nBLOCK_END'))
error_logger.addHandler(error_handler)

# 成功提取日志
success_logger = logging.getLogger('success_logger')
success_handler = logging.FileHandler(SUCCESS_LOG_PATH, mode='w', encoding='utf-8')
success_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - QA_PAIR_START\n%(message)s\nQA_PAIR_END'))
success_logger.addHandler(success_handler)

# --- Prompt 优化 ---
BASE_PROMPT_CHAT = """
你是一个专业的中文问答对提取专家。请从给定的原文中提取**所有**有效的问答对。

🎯 核心任务：
1. 识别所有形式的提问或话题引入
2. 匹配对应的段永平回答
3. 确保问答配对准确完整

📋 提取规则：
1. **问题来源**（多种形式）：
   - 直接提问：网友、问、观众、主持人、Q等开头
   - 文章引用：文章引用、引用、有人说等引出的观点或问题
   - 间接提问：通过描述、举例引出的问题
   - 话题讨论：任何引发段永平回应的内容

2. **答案来源**：仅限段永平、段、大道的回答（排除方丈、其他人）

3. **配对原则**：
   - 一个问题/话题对应一个段永平的回答
   - 问题可以是直接提问，也可以是引用、描述等形式
   - 保持问题和答案的完整性和上下文

4. **输出格式**：JSON数组 [{"question": "问题或话题", "answer": "段永平的回答"}]

❌ 不要提取的内容：
- 方丈的回答
- 其他专家、学者的观点（除非段永平对此有回应）
- 纯描述性文字（除非段永平对此有回应）
- 目录、标题等

✅ 提取示例：

原文1（直接提问）：
网友：什么是stop doing list？
段永平：所谓要做对的事情实际上是通过不做不对的事情来实现的。

原文2（文章引用形式）：
文章引用："微信之父"张小龙就曾说，乔布斯最厉害的地方是他1秒钟就能变成傻瓜。
段：是不是张小龙说的不知道，但这话其实很有道理。

原文3（描述引出）：
有人认为价值投资很难学会。
段：价值投资确实不容易，但有悟性的人可以提高。

输出：
[
  {"question": "什么是stop doing list？", "answer": "所谓要做对的事情实际上是通过不做不对的事情来实现的。"},
  {"question": "文章引用：微信之父张小龙就曾说，乔布斯最厉害的地方是他1秒钟就能变成傻瓜。", "answer": "是不是张小龙说的不知道，但这话其实很有道理。"},
  {"question": "有人认为价值投资很难学会。", "answer": "价值投资确实不容易，但有悟性的人可以提高。"}
]

🔍 请仔细分析以下原文，识别所有引发段永平回应的内容（包括直接提问、文章引用、描述等），并提取所有符合条件的问答对：
"""

# --- 已修改：更智能的后处理函数，优先已知前缀列表，再用通用宽松正则兜底 ---  # 已修改

def clean_question_text(question: str) -> str:  # 已修改
    """去除问题开头的各种发问者前缀（已知列表优先，未知前缀再宽松匹配）"""  # 已修改
    # 安全检查，防止None值
    if question is None:
        return ""
    
    # 确保question是字符串
    question = str(question)
    
    # 已知常见前缀列表  # 已修改
    known_prefixes = ["网友", "记者", "问", "提问者", "主持人", "文章引用", "Q", "观众", "评论", "主持", "用户"]  # 已修改
    pattern = r'^({})[：:]\s*'.format('|'.join(re.escape(p) for p in known_prefixes))  # 已修改
    cleaned = re.sub(pattern, '', question).strip()  # 已修改
    if cleaned != question:  # 已修改
        return cleaned  # 已修改

    # 宽松兜底正则，匹配任何短前缀 + 冒号  # 已修改
    fallback_pattern = r'^[\u4e00-\u9fa5A-Za-z0-9（）【】「」《》''""、,，.。·\s]{1,20}[：:]\s*'  # 已修改
    return re.sub(fallback_pattern, '', question).strip()  # 已修改

# --- PDF 提取 ---
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    all_paragraphs = []
    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            paragraph_text = b[4].replace('\n', ' ').strip()
            if paragraph_text:
                all_paragraphs.append(paragraph_text)
    return "\n\n".join(all_paragraphs)

# --- 改进后的 QA 判断 ---
def block_has_qa(text: str) -> bool:
    """更精确的问答检测，包括多种问题形式"""
    # 直接提问模式
    direct_question_patterns = [
        r"网友[：:]",
        r"问[：:]",
        r"问题[：:]", 
        r"提问[：:]",
        r"主持人[：:]",
        r"观众[：:]",
        r"Q[：:]"
    ]
    
    # 间接问题模式（文章引用、描述等）
    indirect_question_patterns = [
        r"文章引用[：:]",
        r"引用[：:]",
        r"有人说",
        r"有人认为",
        r"有观点认为",
        r"据说",
        r"听说"
    ]
    
    # 段永平的回答
    answer_patterns = [
        r"段永平[：:]",
        r"段[：:]",
        r"大道[：:]"
    ]
    
    has_direct_question = any(re.search(p, text) for p in direct_question_patterns)
    has_indirect_question = any(re.search(p, text) for p in indirect_question_patterns)
    has_answer = any(re.search(p, text) for p in answer_patterns)
    
    # 如果有段永平的回答，且有任何形式的问题引入，就认为包含问答
    return bool(has_answer and (has_direct_question or has_indirect_question))

# --- 预处理问答对识别 ---
def preprocess_qa_text(text: str) -> str:
    """
    预处理文本，标准化问答格式，便于模型识别
    """
    # 标准化问答标识符
    text = re.sub(r'网友[：:]', '网友：', text)
    text = re.sub(r'问[：:]', '问：', text)
    text = re.sub(r'段永平[：:]', '段永平：', text)
    text = re.sub(r'段[：:]', '段：', text)
    text = re.sub(r'大道[：:]', '大道：', text)
    
    # 清理多余空行
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text

# --- 混合分块 ---
def create_hybrid_blocks(text: str, max_size: int, min_size: int) -> List[str]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    all_blocks = []
    i = 0

    while i < len(paragraphs):
        block_paras = []
        current_len = 0

        while i < len(paragraphs):
            para = paragraphs[i]
            block_paras.append(para)
            current_len += len(para)

            if current_len >= max_size:
                i += 1
                break
            i += 1

        block_text = "\n\n".join(block_paras)
        if len(block_text) > min_size:
            all_blocks.append(block_text)

    return all_blocks

# --- 提取 JSON ---
def extract_json(text: str) -> List[dict]:
    """
    提取JSON，支持多种情况：
    1. JSON数组 [{"question": "...", "answer": "..."}, ...]
    2. 单个JSON对象 {"question": "...", "answer": "..."}
    3. 多个独立的JSON对象
    4. 空数组 []
    """
    results = []
    
    try:
        # 先尝试查找```json```包裹的内容
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        
        if json_blocks:
            for json_block in json_blocks:
                results.extend(_parse_json_content(json_block))
        else:
            # 如果没有```json```包裹，直接解析整个文本
            results.extend(_parse_json_content(text))
    
    except Exception as e:
        error_logger.error(f"JSON提取异常: {e}\n原始回复:\n{text}")
    
    # 过滤有效的问答对
    valid_results = []
    for data in results:
        if (isinstance(data, dict) and 
            "question" in data and "answer" in data and
            data.get("question") and data.get("answer") and
            str(data.get("question", "")).strip() and 
            str(data.get("answer", "")).strip()):
            valid_results.append(data)
    
    return valid_results

def _parse_json_content(content: str) -> List[dict]:
    """解析JSON内容，优先处理数组格式，然后处理单个对象和多个对象"""
    results = []
    content = content.strip()
    
    if not content:
        return results
    
    try:
        # 首先尝试解析为JSON（可能是数组或单个对象）
        data = json.loads(content)
        if isinstance(data, list):
            # 如果是数组，直接扩展到结果中
            for item in data:
                if isinstance(item, dict):
                    results.append(item)
        elif isinstance(data, dict):
            # 如果是单个对象，添加到结果中
            results.append(data)
        return results
        
    except json.JSONDecodeError:
        # 如果JSON解析失败，尝试分离多个JSON对象
        try:
            # 查找所有的JSON对象
            json_objects = []
            brace_count = 0
            start_pos = -1
            
            for i, char in enumerate(content):
                if char == '{':
                    if brace_count == 0:
                        start_pos = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_pos != -1:
                        json_str = content[start_pos:i+1]
                        json_objects.append(json_str)
                        start_pos = -1
            
            # 解析每个JSON对象
            for json_str in json_objects:
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        results.append(data)
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            error_logger.error(f"JSON对象分离失败: {e}\n内容:\n{content}")
    
    return results

# --- 调用 Ollama ---
def call_ollama(client: Client, model_name: str, prompt_text: str) -> Optional[str]:
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            options={"temperature": 0.1}
        )
        return response["message"]["content"]
    except Exception as e:
        logging.error(f"Ollama API 调用失败: {e}")
        return None

# --- 主处理流程 ---
def process(pdf_path, output_path, model_name, ratio, max_size, min_size, enable_filter):
    print(f"🔎 开始处理文件: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"📄 文本总长度: {len(raw_text)} 字符")

    print("✂️ 分块中...")
    blocks = create_hybrid_blocks(raw_text, max_size, min_size)
    print(f"✅ 完成分块，共 {len(blocks)} 块")

    if enable_filter:
        blocks = [b for b in blocks if block_has_qa(b)]
        print(f"⚡ 过滤后剩余含问答块: {len(blocks)}")

    if ratio < 1.0:
        sample_size = int(len(blocks) * ratio)
        blocks = blocks[:max(sample_size, 1)]
        print(f"⚡ 顺序采样 {len(blocks)} 块 (比例: {ratio:.0%})")

    if not blocks:
        print("未找到有效块，程序退出。")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        pass

    client = Client(host='http://localhost:11434')
    success_count = 0

    for block in tqdm(blocks, desc="生成问答对中"):
        # 预处理文本
        processed_block = preprocess_qa_text(block)
        prompt = BASE_PROMPT_CHAT + "\n\n" + processed_block

        reply = call_ollama(client, model_name, prompt)

        if reply is None:
            error_logger.error(f"API调用失败或无返回内容。Block:\n{block}")
            continue

        data = extract_json(reply)

        if data:
            for i, d in enumerate(data):
                # 安全检查，确保question和answer不为None
                if not d.get("question") or not d.get("answer"):
                    continue
                    
                clean_q = clean_question_text(d["question"])  # 已修改
                final_data = {
                    "question": clean_q,
                    "answer": d["answer"],
                    "source_text": block
                }
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                
                # 记录成功提取的问答对
                success_log_content = f"""成功提取问答对 #{success_count + i + 1}:

问题: {d["question"]}

答案: {d["answer"]}

原文块:
{block}

{'='*80}"""
                success_logger.info(success_log_content)
            
            success_count += len(data)
            print(f"✅ 从当前块提取了 {len(data)} 个问答对")
        else:
            print(f"❌ 当前块未提取到问答对")
            error_logger.error(f"无法提取有效问答。模型回复: {reply}\n\nBlock:\n{block}")

    print(f"\n🎉 处理完成！共写入 {success_count} 个问答对至 {output_path}")
    print("错误信息已记录到 extraction_errors_final.log 文件，请查看。")

# --- 入口 ---
if __name__ == "__main__":
    if "/path/to/your/file.pdf" in PDF_PATH:
        print("❌ 错误：请先修改 PDF_PATH 为你的 PDF 文件路径。")
    else:
        process(
            pdf_path=PDF_PATH,
            output_path=OUTPUT_PATH,
            model_name=MODEL_NAME,
            ratio=EXTRACT_RATIO,
            max_size=MAX_BLOCK_SIZE,
            min_size=MIN_BLOCK_SIZE,
            enable_filter=ENABLE_QA_FILTER
        )