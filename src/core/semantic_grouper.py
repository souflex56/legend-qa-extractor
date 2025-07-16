# src/core/semantic_grouper.py
"""
智能语义分组器 - 替代原有的机械分块处理器
核心功能：规则预筛选、语义动态分块、领域嵌入模型
整合了SemanticProcessor的功能，提供统一的语义分组接口
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import jieba.posseg as pseg

logger = logging.getLogger(__name__)


class SemanticGrouper:
    """
    智能语义分组器，实现三层分块策略：
    1. 规则预筛选（快速高置信度识别）
    2. 语义动态分块（处理复杂边界、隐性问答）
    3. 领域嵌入模型（对难处理文本精调）
    
    整合了SemanticProcessor的功能，提供更灵活的接口
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
        
        # 🚀 新增：高置信度分组的特殊size限制
        self.high_confidence_min_size = semantic_config.get('high_confidence_min_size', 100)  # 高置信度块的最小size可以更小
        self.high_confidence_max_size = semantic_config.get('high_confidence_max_size', 2000)  # 高置信度块的最大size可以更大
        
        # 🚀 新增：过滤级别配置（来自SemanticProcessor）
        self.filtering_level = config.get('filtering_level', 'balanced')  # strict, balanced, none
        self.semantic_threshold = config.get('semantic_threshold', 0.5)
        
        # 初始化jieba分词器
        self._init_jieba()
        
        # 初始化模型字典
        model_name = semantic_config.get('model_name', 'paraphrase-multilingual-MiniLM-L12-v2')
        device = semantic_config.get('device', None)
        
        try:
            self.models = {
                "general": SentenceTransformer(model_name, device=device),
                # 预留领域模型位置
                # "medical": SentenceTransformer('path/to/biobert'),
                # "financial": SentenceTransformer('path/to/finbert'),
            }
            self.logger.info(f"✅ SentenceTransformer model '{model_name}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"🔥 Failed to load SentenceTransformer model '{model_name}'. Please ensure it is installed and accessible.")
            self.logger.error(f"You may need to run: pip install -U sentence-transformers")
            raise e
        
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
    
    def block_has_qa(self, text: str) -> bool:
        """
        检查文本块是否包含问答模式（来自TextProcessor）
        用于快速识别高置信度的问答段落
        """
        # 检查是否包含问答前缀
        for pattern in self.high_confidence_qa_patterns:
            if re.search(pattern, text):
                return True
        return False
    
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
        改进策略：只拼接严格的一问一答（各自带前缀的单段落）
        其他情况自动划分为中置信度，由LLM处理
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
                # 找到问题，寻找紧邻的答案段落
                qa_content = [para]
                used_indices.add(i)
                j = i + 1
                
                # 寻找紧邻的答案段落（只找下一个段落）
                answer_found = False
                while j < len(paragraphs) and j < i + 3:  # 限制搜索范围，只找紧邻的
                    if j in used_indices:
                        j += 1
                        continue
                        
                    next_para = paragraphs[j].strip()
                    
                    # 🚀 新策略：遇到新问题立即停止
                    is_new_question = any(re.search(pattern, next_para) for pattern in self.question_patterns)
                    if is_new_question:
                        self.logger.debug(f"Found new question at index {j}, stopping current QA collection")
                        break
                    
                    # 检查是否是答案
                    is_answer = any(re.search(pattern, next_para) for pattern in self.answer_patterns)
                    
                    if is_answer:
                        # 🚀 新策略：只拼接严格的一问一答
                        qa_content.append(next_para)
                        used_indices.add(j)
                        answer_found = True
                        
                        # 🚀 不再收集后续内容，保持纯净的一问一答
                        self.logger.debug(f"Found strict QA pair: question at {i}, answer at {j}")
                        break
                    
                    # 如果不是答案，跳过这个段落，继续寻找
                    j += 1
                
                # 如果找到了严格的一问一答，创建高置信度块
                if answer_found:
                    content = "\n\n".join(qa_content)
                    content_size = len(content)
                    
                    # 🚀 使用高置信度分组的特殊size限制
                    if content_size >= self.high_confidence_min_size and content_size <= self.high_confidence_max_size:
                        high_confidence_blocks.append({
                            'content': content,
                            'confidence': 'high',
                            'type': 'rule_based_strict',
                            'indices': sorted(list(used_indices))
                        })
                        self.logger.debug(f"Created strict high-confidence block: {content_size} chars")
                    elif content_size > self.high_confidence_max_size:
                        # 如果超过高置信度最大限制，仍然保留但标记需要分割
                        self.logger.warning(f"High-confidence block too large ({content_size} chars), will be split later")
                        high_confidence_blocks.append({
                            'content': content,
                            'confidence': 'high',
                            'type': 'rule_based_strict',
                            'indices': sorted(list(used_indices))
                        })
                    else:
                        # 🚀 即使小于最小限制，也保留高置信度块（因为严格的一问一答质量很高）
                        if content_size >= 50:  # 绝对最小限制
                            high_confidence_blocks.append({
                                'content': content,
                                'confidence': 'high',
                                'type': 'rule_based_strict_small',
                                'indices': sorted(list(used_indices))
                            })
                            self.logger.debug(f"Created small but strict high-confidence block: {content_size} chars")
                        else:
                            # 太小，标记为未使用，让后续处理
                            self.logger.debug(f"High-confidence block too small ({content_size} chars), skipping")
                            # 创建副本避免在迭代时修改集合
                            indices_to_remove = list(used_indices)
                            for idx in indices_to_remove:
                                used_indices.discard(idx)
                else:
                    # 🚀 新策略：如果没找到严格的一问一答，释放已使用的索引
                    # 让语义分组器处理这种情况
                    self.logger.debug(f"No strict QA pair found for question at index {i}, releasing indices")
                    used_indices.discard(i)
            
            i += 1
        
        # 收集未使用的段落索引
        remaining_indices = [idx for idx in range(len(paragraphs)) if idx not in used_indices]
        
        self.logger.info(f"Rule-based prescreening: {len(high_confidence_blocks)} strict high-confidence blocks found")
        return high_confidence_blocks, remaining_indices
    
    def _group_by_rules_conservative(self, paragraphs: List[str]) -> Tuple[List[str], List[int]]:
        """
        保守的规则分组方法（来自SemanticProcessor）
        更保守的规则：只配对紧邻的问答段落
        """
        groups = []
        used_indices = set()
        
        i = 0
        while i < len(paragraphs):
            if i in used_indices:
                i += 1
                continue

            # 检查当前段落是否包含问答模式
            if self.block_has_qa(paragraphs[i]):
                start_index = i
                
                # 保守规则：将问题与其紧邻的下一个段落配对
                # 如果下一个段落看起来不像另一个问题
                if i + 1 < len(paragraphs) and not self.block_has_qa(paragraphs[i + 1]):
                    # 这可能是问答对
                    end_index = i + 2
                    group_paragraphs = paragraphs[start_index:end_index]
                    groups.append("\n\n".join(group_paragraphs))
                    used_indices.add(i)
                    used_indices.add(i + 1)
                    i += 2
                else:
                    # 这可能是独立块或问题后跟另一个问题，作为单段落组处理
                    end_index = i + 1
                    group_paragraphs = paragraphs[start_index:end_index]
                    groups.append("\n\n".join(group_paragraphs))
                    used_indices.add(i)
                    i += 1
            else:
                i += 1
        
        remaining_indices = sorted([idx for idx in range(len(paragraphs)) if idx not in used_indices])
        self.logger.info(f"Identified {len(groups)} high-confidence groups using conservative rules.")
        return groups, remaining_indices
    
    def _group_by_semantics_simple(self, paragraphs: List[str], max_question_len: int, threshold: float) -> Tuple[List[str], List[str]]:
        """
        简化的语义分组方法（来自SemanticProcessor）
        使用sklearn的cosine_similarity进行语义相似度计算
        """
        if not paragraphs:
            return [], []
            
        groups = []
        used_indices = set()
        
        # 1. 识别潜在问题
        potential_questions = {}
        for i, p in enumerate(paragraphs):
            if len(p) <= max_question_len:
                # 简单启发式：可能是问题
                potential_questions[i] = p
        
        if not potential_questions:
            return [], paragraphs

        # 2. 编码所有段落进行语义比较
        model = self.models["general"]
        embeddings = model.encode(paragraphs, convert_to_tensor=True)

        # 3. 遍历潜在问题并找到相关答案
        for q_index, q_text in potential_questions.items():
            if q_index in used_indices:
                continue

            current_group_indices = {q_index}
            q_embedding = embeddings[q_index].cpu().numpy().reshape(1, -1)

            # 向前看后续段落
            for a_index in range(q_index + 1, len(paragraphs)):
                if a_index in used_indices:
                    # 如果遇到已经是另一个组的一部分的段落，停止
                    break
                
                a_embedding = embeddings[a_index].cpu().numpy().reshape(1, -1)
                
                # 计算问题和潜在答案之间的相似度
                similarity = cosine_similarity(q_embedding, a_embedding)[0][0]
                
                if similarity >= threshold:
                    # 这个段落语义相关，添加到组中
                    current_group_indices.add(a_index)
                else:
                    # 语义链接断开，停止形成这个组
                    break
            
            if len(current_group_indices) > 1:  # 找到有效组（问题+至少一个答案）
                # 排序索引以保持原始顺序
                sorted_indices = sorted(list(current_group_indices))
                group_paras = [paragraphs[i] for i in sorted_indices]
                groups.append("\n\n".join(group_paras))
                used_indices.update(sorted_indices)

        # 4. 收集所有未分组的段落
        ungrouped_paras = [p for i, p in enumerate(paragraphs) if i not in used_indices]
        
        self.logger.info(f"Identified {len(groups)} medium-confidence groups using semantics.")
        return groups, ungrouped_paras
    
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
        使用简化的跟踪器记录处理过程
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
    
    def group_text_by_semantics(self, text: str, config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        兼容SemanticProcessor的接口方法
        处理原始文本，返回语义分组结果
        
        Args:
            text: 原始文本
            config: 配置参数（可选，会与初始化时的配置合并）
            
        Returns:
            分组结果列表，每个元素包含content和type
        """
        if not text:
            return []
        
        # 合并配置
        if config:
            merged_config = self.config.copy()
            merged_config.update(config)
            # 临时更新实例变量
            self.filtering_level = merged_config.get('filtering_level', self.filtering_level)
            self.semantic_threshold = merged_config.get('semantic_threshold', self.semantic_threshold)
        else:
            merged_config = self.config
        
        # 1. 分割文本为段落
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # 2. 第一轮：使用保守规则识别高置信度组
        rule_based_groups, remaining_indices = self._group_by_rules_conservative(paragraphs)
        
        # 3. 处理不同的过滤级别
        if self.filtering_level == 'strict':
            self.logger.info(f"Filtering level is 'strict'. Returning {len(rule_based_groups)} high-confidence groups.")
            return [{"content": group, "type": "high-confidence"} for group in rule_based_groups]
        
        # 对于 'balanced' 和 'none'，需要处理剩余段落
        remaining_paragraphs = [paragraphs[i] for i in remaining_indices]
        
        # 4. 第二轮：使用语义相似度识别中置信度组
        semantic_groups, ungrouped_paras = self._group_by_semantics_simple(
            remaining_paragraphs,
            merged_config.get('max_question_length', self.max_question_length),
            merged_config.get('semantic_threshold', self.semantic_threshold)
        )
        
        # 5. 合并和最终化组
        all_groups = []
        all_groups.extend([{"content": group, "type": "high-confidence"} for group in rule_based_groups])
        all_groups.extend([{"content": group, "type": "medium-confidence"} for group in semantic_groups])
        
        if self.filtering_level == 'none':
            # 添加所有剩余段落作为低置信度组
            all_groups.extend([{"content": para, "type": "low-confidence"} for para in ungrouped_paras])
        
        self.logger.info(f"Total groups created: {len(all_groups)} (High: {len(rule_based_groups)}, Medium: {len(semantic_groups)}, Low: {len(ungrouped_paras) if self.filtering_level == 'none' else 0})")
        return all_groups 