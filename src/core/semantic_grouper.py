# src/core/semantic_grouper.py
"""
智能语义分组器 - 替代原有的机械分块处理器
核心功能：规则预筛选、语义动态分块、领域嵌入模型
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
import jieba
import jieba.posseg as pseg

logger = logging.getLogger(__name__)


class SemanticGrouper:
    """
    智能语义分组器，实现三层分块策略：
    1. 规则预筛选（快速高置信度识别）
    2. 语义动态分块（处理复杂边界、隐性问答）
    3. 领域嵌入模型（对难处理文本精调）
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        
        # 从配置中提取语义分组相关配置
        semantic_config = config.get('semantic_grouping', {})
        self.max_question_length = semantic_config.get('max_question_length', 50)
        self.default_similarity_threshold = semantic_config.get('default_similarity_threshold', 0.65)
        self.std_factor = semantic_config.get('std_factor', 0.5)
        
        # 分块大小配置
        self.max_block_size = config.get('max_block_size', 1500)
        self.min_block_size = config.get('min_block_size', 200)
        
        # 初始化jieba分词器
        self._init_jieba()
        
        # 初始化模型字典
        self.models = {
            "general": SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'),  # 更轻量的多语言模型
            # 预留领域模型位置
            # "medical": SentenceTransformer('path/to/biobert'),
            # "financial": SentenceTransformer('path/to/finbert'),
        }
        
        # 高置信度问答模式（规则预筛选）
        # 分为问题模式和答案模式
        self.question_patterns = [
            r"网友[：:]",
            r"记者[：:]",
            r"问[：:]",
            r"问题[：:]",
            r"提问[：:]",
            r"主持人[：:]",
            r"观众[：:]",
            r"Q[：:]"
        ]
        
        self.answer_patterns = [
            r"段永平[：:]",
            r"段[：:]",
            r"A[：:]",
            r"答[：:]",
            r"回答[：:]"
        ]
        
        # 所有QA模式（用于兼容性）
        self.high_confidence_qa_patterns = self.question_patterns + self.answer_patterns
        
        # 潜在问题模式
        self.potential_question_patterns = [
            r".*[？?]$",  # 以问号结尾
            r"^(什么|为什么|怎么|如何|是否|有没有|能不能)",  # 疑问词开头
            r"(是什么|为什么|怎么样|如何|吗|呢)[？?]?$",  # 疑问词结尾
        ]
        
        self.logger.info("Semantic Grouper initialized successfully")
    
    def _init_jieba(self):
        """初始化jieba分词器，添加自定义词典"""
        # 添加一些金融/投资相关的专有名词
        custom_words = [
            "价值投资", "stop doing list", "能力圈", "护城河", 
            "复利", "市盈率", "现金流", "ROE", "段永平"
        ]
        for word in custom_words:
            jieba.add_word(word)
    
    def _detect_domain(self, text_chunk: str) -> str:
        """检测文本所属领域，用于选择合适的嵌入模型"""
        # 简化版领域检测
        medical_keywords = ["药品", "患者", "治疗", "症状", "医院", "医生", "疾病"]
        financial_keywords = ["$", "收益率", "投资", "股票", "基金", "财报", "市值", "估值"]
        
        text_lower = text_chunk.lower()
        
        # 统计关键词出现次数
        medical_count = sum(1 for keyword in medical_keywords if keyword in text_chunk)
        financial_count = sum(1 for keyword in financial_keywords if keyword in text_chunk)
        
        # 根据关键词密度判断领域
        if medical_count >= 2:
            return "medical"
        elif financial_count >= 2:
            return "financial"
        else:
            return "general"
    
    def _is_potential_question(self, paragraph: str) -> bool:
        """判断段落是否为潜在问题"""
        # 长度检查
        if len(paragraph) > self.max_question_length:
            return False
        
        # 检查是否匹配潜在问题模式
        for pattern in self.potential_question_patterns:
            if re.search(pattern, paragraph.strip()):
                return True
        
        # 使用jieba进行词性分析
        words = pseg.cut(paragraph)
        has_question_word = False
        
        for word, flag in words:
            # 检查疑问代词(r)、疑问副词(ry)等
            if flag in ['r', 'ry'] and word in ['什么', '怎么', '为什么', '哪里', '哪个', '如何', '是否']:
                has_question_word = True
                break
        
        return has_question_word
    
    def _calculate_dynamic_threshold(self, paragraphs: List[str], model: SentenceTransformer) -> float:
        """动态计算相似度阈值"""
        if len(paragraphs) < 3:
            return self.default_similarity_threshold
        
        # 编码文本获取嵌入向量
        embeddings = model.encode(paragraphs, convert_to_tensor=True)
        
        # 计算相似度矩阵
        sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        
        # 获取上三角矩阵（排除对角线）
        upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
        similarities = sim_matrix[upper_triangle_indices]
        
        if len(similarities) == 0:
            return self.default_similarity_threshold
        
        # 计算平均值和标准差
        window_avg = np.mean(similarities)
        window_std = np.std(similarities)
        
        # 动态阈值 = 平均值 - 因子 * 标准差
        # 这样可以自适应不同文本的相似度分布
        threshold = window_avg - self.std_factor * window_std
        
        # 限制在合理范围内
        return np.clip(threshold, 0.5, 0.9)
    
    def _rule_based_prescreening(self, paragraphs: List[str]) -> Tuple[List[Dict], List[int]]:
        """
        规则预筛选：快速识别高置信度问答对
        改进逻辑：专门识别问题-答案配对
        返回：(高置信度块列表, 需要进一步处理的段落索引列表)
        """
        high_confidence_blocks = []
        used_indices = set()
        
        i = 0
        while i < len(paragraphs):
            if i in used_indices:
                i += 1
                continue
                
            para = paragraphs[i].strip()
            
            # 检查是否是问题开始
            is_question = any(re.search(pattern, para) for pattern in self.question_patterns)
            
            if is_question:
                # 找到问题，寻找对应的答案
                qa_content = [para]
                used_indices.add(i)
                j = i + 1
                
                # 寻找答案段落
                answer_found = False
                while j < len(paragraphs) and j < i + 5:  # 限制搜索范围，避免过度贪婪
                    if j in used_indices:
                        j += 1
                        continue
                        
                    next_para = paragraphs[j].strip()
                    
                    # 检查是否是答案
                    is_answer = any(re.search(pattern, next_para) for pattern in self.answer_patterns)
                    
                    if is_answer:
                        qa_content.append(next_para)
                        used_indices.add(j)
                        answer_found = True
                        
                        # 继续收集答案的后续内容，但要更谨慎
                        k = j + 1
                        while k < len(paragraphs) and k < j + 2:  # 限制为最多1个后续段落
                            if k in used_indices:
                                k += 1
                                continue
                                
                            next_next_para = paragraphs[k].strip()
                            is_new_qa = any(re.search(pattern, next_next_para) for pattern in self.high_confidence_qa_patterns)
                            
                            if is_new_qa:
                                break
                            
                            # 只有当段落看起来像是答案的延续时才添加
                            if len(next_next_para) > 20 and not next_next_para.endswith('？'):
                                qa_content.append(next_next_para)
                                used_indices.add(k)
                            k += 1
                        break
                    
                    # 如果不是答案，但也不是新问题，可能是问题的补充
                    is_new_question = any(re.search(pattern, next_para) for pattern in self.question_patterns)
                    if is_new_question:
                        break
                    
                    j += 1
                
                # 如果找到了完整的问答对，创建高置信度块
                if answer_found:
                    content = "\n\n".join(qa_content)
                    if len(content) >= self.min_block_size:
                        high_confidence_blocks.append({
                            'content': content,
                            'confidence': 'high',
                            'type': 'rule_based',
                            'indices': sorted(list(used_indices))
                        })
            
            i += 1
        
        # 收集未使用的段落索引
        remaining_indices = [idx for idx in range(len(paragraphs)) if idx not in used_indices]
        
        self.logger.info(f"Rule-based prescreening: {len(high_confidence_blocks)} high-confidence blocks found")
        return high_confidence_blocks, remaining_indices
    
    def _semantic_dynamic_grouping(self, paragraphs: List[str], indices: List[int]) -> List[Dict]:
        """语义动态分组：处理复杂边界和隐性问答"""
        if not indices:
            return []
        
        semantic_blocks = []
        
        # 创建滑动窗口进行分析
        window_size = min(5, len(indices))  # 窗口大小
        
        i = 0
        while i < len(indices):
            # 获取当前窗口的段落
            window_end = min(i + window_size, len(indices))
            window_indices = indices[i:window_end]
            window_paragraphs = [paragraphs[idx] for idx in window_indices]
            
            # 检测领域并选择模型
            combined_text = " ".join(window_paragraphs)
            domain = self._detect_domain(combined_text)
            model = self.models.get(domain, self.models["general"])
            
            # 计算动态阈值
            if len(window_paragraphs) >= 3:
                threshold = self._calculate_dynamic_threshold(window_paragraphs, model)
            else:
                threshold = self.default_similarity_threshold
            
            # 检查第一个段落是否为潜在问题
            first_para = window_paragraphs[0]
            if self._is_potential_question(first_para) and len(window_paragraphs) > 1:
                # 潜在问题检测到，尝试找到对应答案
                question_embedding = model.encode(first_para, convert_to_tensor=True)
                
                # 计算与后续段落的相似度
                best_match_idx = -1
                best_similarity = 0
                
                for j in range(1, len(window_paragraphs)):
                    answer_embedding = model.encode(window_paragraphs[j], convert_to_tensor=True)
                    similarity = util.cos_sim(question_embedding, answer_embedding).item()
                    
                    if similarity > threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match_idx = j
                
                if best_match_idx > 0:
                    # 找到匹配的答案，创建语义块
                    qa_indices = window_indices[0:best_match_idx+1]
                    qa_content = "\n\n".join([paragraphs[idx] for idx in qa_indices])
                    
                    semantic_blocks.append({
                        'content': qa_content,
                        'confidence': 'medium',
                        'type': 'semantic',
                        'similarity_score': best_similarity,
                        'domain': domain,
                        'indices': qa_indices
                    })
                    
                    # 跳过已处理的段落
                    i += best_match_idx + 1
                    continue
            
            # 如果不是问答对，尝试基于语义相似度合并段落
            current_group = [window_paragraphs[0]]
            current_indices = [window_indices[0]]
            
            if len(window_paragraphs) > 1:
                base_embedding = model.encode(window_paragraphs[0], convert_to_tensor=True)
                
                for j in range(1, len(window_paragraphs)):
                    para_embedding = model.encode(window_paragraphs[j], convert_to_tensor=True)
                    similarity = util.cos_sim(base_embedding, para_embedding).item()
                    
                    if similarity > threshold:
                        current_group.append(window_paragraphs[j])
                        current_indices.append(window_indices[j])
                    else:
                        break
            
            # 创建语义块
            if len("\n\n".join(current_group)) >= self.min_block_size:
                semantic_blocks.append({
                    'content': "\n\n".join(current_group),
                    'confidence': 'low',
                    'type': 'semantic_merge',
                    'domain': domain,
                    'indices': current_indices
                })
            
            i += len(current_group)
        
        self.logger.info(f"Semantic grouping: {len(semantic_blocks)} semantic blocks created")
        return semantic_blocks
    
    def _merge_and_optimize_blocks(self, all_blocks: List[Dict]) -> List[Dict]:
        """合并和优化所有块，确保大小合适"""
        # 按原始索引排序
        sorted_blocks = sorted(all_blocks, key=lambda x: x['indices'][0])
        
        optimized_blocks = []
        current_merged = None
        
        for block in sorted_blocks:
            block_size = len(block['content'])
            
            # 如果块太大，需要分割
            if block_size > self.max_block_size:
                if current_merged:
                    optimized_blocks.append(current_merged)
                    current_merged = None
                
                # 分割超大块
                split_blocks = self._split_large_block(block)
                optimized_blocks.extend(split_blocks)
            
            # 如果块太小，尝试合并
            elif block_size < self.min_block_size:
                if current_merged:
                    # 检查是否可以合并
                    merged_size = len(current_merged['content']) + block_size + 4  # +4 for \n\n
                    if merged_size <= self.max_block_size:
                        current_merged['content'] += "\n\n" + block['content']
                        current_merged['indices'].extend(block['indices'])
                        # 更新置信度为最低值
                        if current_merged['confidence'] == 'high' and block['confidence'] != 'high':
                            current_merged['confidence'] = block['confidence']
                    else:
                        optimized_blocks.append(current_merged)
                        current_merged = block
                else:
                    current_merged = block
            
            # 块大小合适
            else:
                if current_merged:
                    # 检查是否可以合并
                    merged_size = len(current_merged['content']) + block_size + 4
                    if merged_size <= self.max_block_size:
                        current_merged['content'] += "\n\n" + block['content']
                        current_merged['indices'].extend(block['indices'])
                        if current_merged['confidence'] == 'high' and block['confidence'] != 'high':
                            current_merged['confidence'] = block['confidence']
                    else:
                        optimized_blocks.append(current_merged)
                        current_merged = None
                        optimized_blocks.append(block)
                else:
                    optimized_blocks.append(block)
        
        # 处理最后的块
        if current_merged:
            optimized_blocks.append(current_merged)
        
        return optimized_blocks
    
    def _split_large_block(self, block: Dict) -> List[Dict]:
        """分割超大块"""
        content = block['content']
        paragraphs = content.split('\n\n')
        
        split_blocks = []
        current_content = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size + 4 > self.max_block_size:
                if current_content:
                    split_blocks.append({
                        'content': '\n\n'.join(current_content),
                        'confidence': block['confidence'],
                        'type': block['type'] + '_split',
                        'indices': []  # 分割后难以追踪原始索引
                    })
                current_content = [para]
                current_size = para_size
            else:
                current_content.append(para)
                current_size += para_size + 4
        
        if current_content:
            split_blocks.append({
                'content': '\n\n'.join(current_content),
                'confidence': block['confidence'],
                'type': block['type'] + '_split',
                'indices': []
            })
        
        return split_blocks
    
    def group(self, paragraphs: List[str]) -> List[Dict[str, Any]]:
        """
        主分组方法：综合运用三层策略
        1. 规则预筛选
        2. 语义动态分组
        3. 块优化和合并
        """
        if not paragraphs:
            return []
        
        self.logger.info(f"Starting semantic grouping for {len(paragraphs)} paragraphs")
        
        # 第一层：规则预筛选
        high_confidence_blocks, remaining_indices = self._rule_based_prescreening(paragraphs)
        
        # 第二层：语义动态分组（处理剩余段落）
        semantic_blocks = []
        if remaining_indices:
            semantic_blocks = self._semantic_dynamic_grouping(paragraphs, remaining_indices)
        
        # 合并所有块
        all_blocks = high_confidence_blocks + semantic_blocks
        
        # 第三层：优化和合并
        optimized_blocks = self._merge_and_optimize_blocks(all_blocks)
        
        self.logger.info(f"Semantic grouping completed: {len(optimized_blocks)} final blocks")
        
        # 添加统计信息
        high_conf_count = sum(1 for b in optimized_blocks if b.get('confidence') == 'high')
        medium_conf_count = sum(1 for b in optimized_blocks if b.get('confidence') == 'medium')
        low_conf_count = sum(1 for b in optimized_blocks if b.get('confidence') == 'low')
        
        self.logger.info(f"Block confidence distribution - High: {high_conf_count}, Medium: {medium_conf_count}, Low: {low_conf_count}")
        
        return optimized_blocks 