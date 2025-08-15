# src/core/semantic_grouper.py
"""
智能语义分组器 - 替代原有的机械分块处理器
核心功能：规则预筛选、语义动态分块、领域嵌入模型
整合了SemanticProcessor的功能，提供统一的语义分组接口
"""

import re
import logging
import numpy as np
import yaml
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
        
        # 🔥 加载目标人物配置
        self.target_person = self._load_target_person_config(config)
        
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
        
        # 🔥 动态生成问答模式识别范围
        self.question_patterns = self._generate_question_patterns()
        self.answer_patterns = self._generate_answer_patterns()
        
        # 所有QA模式（用于兼容性）
        self.high_confidence_qa_patterns = self.question_patterns + self.answer_patterns
        
        # 🔥 增强潜在问题模式
        self.potential_question_patterns = [
            r".*[？?]$",  # 以问号结尾
            r"^(什么|为什么|怎么|如何|是否|有没有|能不能)",  # 疑问词开头
            r"(是什么|为什么|怎么样|如何|吗|呢)[？?]?$",  # 疑问词结尾
            # 🚀 新增：更多问句模式
            r".*想问.*",
            r".*请教.*",
            r".*想了解.*",
            r".*求解.*",
        ]
        
        # 🔥 添加动态模式生成的日志信息
        target_name = self.target_person.get('main_name', 'Unknown') if self.target_person else 'Default'
        questioner_count = len(self.target_person.get('questioner_types', [])) if self.target_person else 0
        self.logger.info(f"Semantic Grouper initialized successfully:")
        self.logger.info(f"  - Target person: {target_name}")
        self.logger.info(f"  - Question patterns: {len(self.question_patterns)} generated")
        self.logger.info(f"  - Answer patterns: {len(self.answer_patterns)} generated")
        self.logger.info(f"  - Questioner types: {questioner_count} configured")
    
    def _load_target_person_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        🔥 加载目标人物配置
        支持多种加载方式：
        1. 直接传入target_person字典（向后兼容）
        2. 通过target_person_config路径加载YAML文件
        """
        # 如果配置中已经有target_person字典，直接使用（向后兼容）
        if 'target_person' in config and config['target_person']:
            self.logger.info("Using target_person config from dictionary")
            return config['target_person']
        
        # 如果有target_person_config路径，从YAML文件加载
        person_config_path = config.get('target_person_config')
        if person_config_path:
            try:
                self.logger.info(f"Loading target person config from: {person_config_path}")
                with open(person_config_path, 'r', encoding='utf-8') as f:
                    person_config = yaml.safe_load(f)
                
                # 提取target_person部分
                if 'target_person' in person_config:
                    target_person = person_config['target_person'].copy()
                    # 同时添加questioner_types信息（如果存在）
                    if 'questioner_types' in person_config:
                        target_person['questioner_types'] = person_config['questioner_types']
                    return target_person
                else:
                    self.logger.warning(f"No 'target_person' found in {person_config_path}")
                    return {}
            except Exception as e:
                self.logger.error(f"Failed to load target person config from {person_config_path}: {e}")
                return {}
        
        # 如果都没有，返回空字典
        self.logger.warning("No target person configuration found, using defaults")
        return {}
    
    def _init_jieba(self):
        """初始化jieba分词器，添加自定义词典"""
        # 添加一些金融/投资相关的专有名词
        custom_words = [
            "价值投资", "stop doing list", "能力圈", "护城河", 
            "复利", "市盈率", "现金流", "ROE"
        ]
        
        # 🔥 从配置中添加目标人物相关词汇
        if self.target_person:
            main_name = self.target_person.get('main_name', '')
            aliases = self.target_person.get('aliases', [])
            
            if main_name:
                custom_words.append(main_name)
            
            custom_words.extend(aliases)
        
        for word in custom_words:
            jieba.add_word(word)
    
    def _generate_question_patterns(self) -> List[str]:
        """
        🔥 根据配置动态生成问题模式
        """
        # 基础问题标识符
        base_question_identifiers = [
            "问", "问题", "提问", "Q"
        ]
        
        # 从配置中获取提问者类型
        questioner_types = []
        if self.target_person:
            questioner_types = self.target_person.get('questioner_types', [])
        
        # 如果配置中没有定义，使用默认的
        if not questioner_types:
            questioner_types = ["网友", "记者", "投资者", "主持人", "观众"]
        
        patterns = []
        
        # 为每种提问者类型生成模式
        for questioner in questioner_types:
            # 基础模式
            patterns.append(rf"{questioner}[：:]")
            # 编号前缀模式
            patterns.append(rf"\d+\.\s*{questioner}[：:]")
            # 带标识符模式
            patterns.append(rf"{questioner}[A-Za-z0-9]*[：:]")
        
        # 添加基础问题标识符
        for identifier in base_question_identifiers:
            patterns.append(rf"{identifier}[：:]")
            patterns.append(rf"\d+\.\s*{identifier}[：:]")
        
        # 添加间接引用模式
        indirect_patterns = [
            r"有人问[：:]",
            r"有朋友问[：:]",
            r"有网友问[：:]",
            r"文章引用[：:]",
            r"引用[：:]",
            r"有人说[：:]",
            r"有观点认为[：:]"
        ]
        patterns.extend(indirect_patterns)
        
        # 添加自然语言问句模式
        patterns.append(r".*[？?]$")  # 以问号结尾
        
        return patterns
    
    def _generate_answer_patterns(self) -> List[str]:
        """
        🔥 根据配置动态生成答案模式
        """
        patterns = []
        
        # 基础答案标识符
        base_answer_identifiers = ["A", "答", "回答"]
        
        # 从配置中获取目标人物信息
        if self.target_person:
            main_name = self.target_person.get('main_name', '')
            aliases = self.target_person.get('aliases', [])
            
            # 为主要名字生成模式
            if main_name:
                patterns.append(rf"{main_name}[：:]")
                patterns.append(rf"\d+\.\s*{main_name}[：:]")
                # 支持名字中间有空格的情况
                spaced_name = r"\s*".join(main_name)
                patterns.append(rf"{spaced_name}[：:]")
            
            # 为别名生成模式
            for alias in aliases:
                patterns.append(rf"{alias}[：:]")
                patterns.append(rf"\d+\.\s*{alias}[：:]")
        
        # 添加基础答案标识符
        for identifier in base_answer_identifiers:
            patterns.append(rf"{identifier}[：:]")
            patterns.append(rf"\d+\.\s*{identifier}[：:]")
        
        # 如果没有配置目标人物，使用默认的段永平模式（向后兼容）
        if not patterns or not self.target_person:
            fallback_patterns = [
                r"段永平[：:]",
                r"段[：:]",
                r"大道[：:]",
                r"老段[：:]",
                r"\d+\.\s*段永平[：:]",
                r"\d+\.\s*段[：:]",
                r"段\s*永\s*平[：:]"
            ]
            patterns.extend(fallback_patterns)
        
        return patterns
    
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
        🔥 改进策略：更宽松的问答匹配，支持不完美格式
        1. 扩大搜索范围，不仅限于严格的一问一答
        2. 支持编号前缀和格式变体
        3. 允许问题和答案之间有间隔段落
        返回：(高置信度块列表, 需要进一步处理的段落索引列表)
        """
        high_confidence_blocks = []
        used_indices = set()
        
        self.logger.info(f"🔍 Starting rule-based prescreening for {len(paragraphs)} paragraphs")
        
        i = 0
        while i < len(paragraphs):
            if i in used_indices:
                i += 1
                continue
                
            para = paragraphs[i].strip()
            
            # 🔥 改进：更灵活的问题识别
            is_question = self._is_flexible_question(para)
            
            if is_question:
                self.logger.debug(f"Found potential question at index {i}: {para[:50]}...")
                
                # 🔥 扩大搜索范围：向前找最多5个段落
                qa_content = [para]
                qa_indices = [i]
                used_indices.add(i)
                
                # 🔥 更宽松的答案搜索策略
                answer_found = False
                search_range = min(len(paragraphs), i + 6)  # 扩大搜索范围到5个段落
                
                for j in range(i + 1, search_range):
                    if j in used_indices:
                        continue
                        
                    next_para = paragraphs[j].strip()
                    
                    # 检查是否是新问题（如果是，停止搜索）
                    if self._is_flexible_question(next_para) and j > i + 1:
                        self.logger.debug(f"Found new question at index {j}, stopping search")
                        break
                    
                    # 🔥 更灵活的答案识别
                    is_answer = self._is_flexible_answer(next_para)
                    
                    if is_answer:
                        # 找到答案，收集这个答案
                        qa_content.append(next_para)
                        qa_indices.append(j)
                        used_indices.add(j)
                        answer_found = True
                        self.logger.debug(f"Found answer at index {j}: {next_para[:50]}...")
                        
                        # 🔥 新策略：继续收集后续的相关答案段落
                        # 检查下一个段落是否是同一回答者的补充
                        k = j + 1
                        while k < search_range and k < j + 3:  # 最多再收集2个相关段落
                            if k in used_indices:
                                k += 1
                                continue
                                
                            next_next_para = paragraphs[k].strip()
                            
                            # 如果遇到新问题，停止
                            if self._is_flexible_question(next_next_para):
                                break
                            
                            # 如果是明显的答案补充（不带前缀但相关），则收集
                            if (not self._is_flexible_answer(next_next_para) and 
                                not self._is_flexible_question(next_next_para) and
                                len(next_next_para) > 20):  # 有一定长度的非问答段落
                                qa_content.append(next_next_para)
                                qa_indices.append(k)
                                used_indices.add(k)
                                self.logger.debug(f"Collected supplement at index {k}: {next_next_para[:30]}...")
                            else:
                                break
                            k += 1
                        break
                    
                    # 🔥 如果段落很短且不是明显的问题/答案，可能是中间的描述文字，收集它
                    elif len(next_para) < 100 and not self._is_flexible_question(next_para):
                        qa_content.append(next_para)
                        qa_indices.append(j)
                        used_indices.add(j)
                        self.logger.debug(f"Collected intermediate text at index {j}: {next_para[:30]}...")
                
                # 创建问答块（无论是否找到严格的答案）
                if len(qa_content) > 1 or answer_found:  # 至少有问题+其他内容，或明确找到答案
                    content = "\n\n".join(qa_content)
                    content_size = len(content)
                    
                    # 🔥 更宽松的块大小判断
                    confidence = 'high' if answer_found else 'medium'
                    block_type = 'rule_based_strict' if answer_found else 'rule_based_loose'
                    
                    if content_size >= 50:  # 降低最小限制
                        high_confidence_blocks.append({
                            'content': content,
                            'confidence': confidence,
                            'type': block_type,
                            'indices': sorted(qa_indices),
                            'has_answer': answer_found
                        })
                        self.logger.debug(f"Created {confidence} confidence block: {content_size} chars, indices {qa_indices}")
                    else:
                        # 太小，释放索引让后续处理
                        self.logger.debug(f"Block too small ({content_size} chars), releasing indices")
                        for idx in qa_indices:
                            used_indices.discard(idx)
                else:
                    # 只有问题没有其他内容，释放索引
                    self.logger.debug(f"Only question found, no additional content, releasing index {i}")
                    used_indices.discard(i)
            
            i += 1
        
        # 收集未使用的段落索引
        remaining_indices = [idx for idx in range(len(paragraphs)) if idx not in used_indices]
        
        self.logger.info(f"Rule-based prescreening completed:")
        self.logger.info(f"  - High/Medium confidence blocks: {len(high_confidence_blocks)}")
        self.logger.info(f"  - Remaining paragraphs for semantic processing: {len(remaining_indices)}")
        
        return high_confidence_blocks, remaining_indices
    
    def _is_flexible_question(self, paragraph: str) -> bool:
        """
        🔥 更灵活的问题识别
        支持多种格式变体和间接问法
        """
        if not paragraph:
            return False
        
        # 清理段落（移除编号前缀）
        clean_para = re.sub(r'^\d+\.\s*', '', paragraph).strip()
        
        # 检查是否匹配问题模式
        for pattern in self.question_patterns:
            if re.search(pattern, paragraph):
                return True
        
        # 🔥 新增：检查是否是自然语言问句
        if re.search(r'[？?]$', clean_para):
            return True
            
        # 🔥 新增：检查疑问词开头
        question_words = ['什么', '为什么', '怎么', '如何', '是否', '有没有', '能不能', '会不会', '应该', '想问', '请教']
        for word in question_words:
            if clean_para.startswith(word) or f'想{word}' in clean_para:
                return True
        
        return False
    
    def _is_flexible_answer(self, paragraph: str) -> bool:
        """
        🔥 更灵活的答案识别
        支持多种回答者标识和格式变体
        """
        if not paragraph:
            return False
        
        # 检查是否匹配答案模式
        for pattern in self.answer_patterns:
            if re.search(pattern, paragraph):
                return True
        
        return False
    
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

    def _improved_semantic_grouping(self, paragraphs: List[str], indices: List[int]) -> List[Dict]:
        """
        改进的语义分组方法：解决段落丢失问题
        核心改进：
        1. 使用全局相似度矩阵，避免窗口边界问题
        2. 采用层次聚类思想，确保相关段落被分到同一组
        3. 动态调整分组策略，优先保证内容完整性
        """
        if not indices:
            return []
        
        semantic_blocks = []
        model = self.models["general"]  # 使用通用模型
        
        # 计算所有段落之间的相似度矩阵
        paragraph_texts = [paragraphs[idx] for idx in indices]
        embeddings = model.encode(paragraph_texts, convert_to_tensor=True)
        similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        
        # 计算动态阈值
        if len(paragraph_texts) >= 3:
            threshold = self._calculate_dynamic_threshold(paragraph_texts, model)
        else:
            threshold = self.default_similarity_threshold
        
        # 使用改进的分组策略
        i = 0
        while i < len(indices):
            current_group = [paragraph_texts[0]]  # 从当前段落开始
            current_indices = [indices[0]]
            current_embeddings = [embeddings[0]]
            
            # 向前扩展：寻找与当前组高度相似的段落
            j = i + 1
            while j < len(indices):
                # 计算当前段落与组内所有段落的平均相似度
                current_embedding = embeddings[j]
                similarities = []
                
                for group_embedding in current_embeddings:
                    sim = util.cos_sim(current_embedding.unsqueeze(0), group_embedding.unsqueeze(0)).item()
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                
                # 如果平均相似度超过阈值，加入当前组
                if avg_similarity > threshold:
                    current_group.append(paragraph_texts[j])
                    current_indices.append(indices[j])
                    current_embeddings.append(current_embedding)
                    j += 1
                else:
                    # 检查是否为潜在问题-答案对
                    if self._is_potential_question(paragraph_texts[j]):
                        # 尝试找到对应的答案段落
                        answer_found = False
                        for k in range(j + 1, min(j + 4, len(indices))):  # 在后续3个段落中寻找答案
                            answer_sim = util.cos_sim(current_embedding.unsqueeze(0), embeddings[k].unsqueeze(0)).item()
                            if answer_sim > threshold * 0.8:  # 稍微降低阈值
                                # 将问题和答案都加入当前组
                                current_group.extend(paragraph_texts[j:k+1])
                                current_indices.extend(indices[j:k+1])
                                current_embeddings.extend([embeddings[j], embeddings[k]])
                                j = k + 1
                                answer_found = True
                                break
                        
                        if not answer_found:
                            break
                    else:
                        break
            
            # 创建语义块
            if len("\n\n".join(current_group)) >= self.min_block_size:
                semantic_blocks.append({
                    'content': "\n\n".join(current_group),
                    'confidence': 'medium',
                    'type': 'improved_semantic',
                    'domain': self._detect_domain("\n\n".join(current_group)),
                    'indices': current_indices
                })
            
            i = j  # 移动到下一个未处理的段落
        
        self.logger.info(f"Improved semantic grouping: {len(semantic_blocks)} semantic blocks created")
        return semantic_blocks

    def _hierarchical_semantic_grouping(self, paragraphs: List[str], indices: List[int]) -> List[Dict]:
        """
        层次聚类语义分组：更智能的分组策略
        1. 首先识别主题边界（问题段落）
        2. 然后基于语义相似度将相关段落聚类
        3. 确保每个主题的完整性
        """
        if not indices:
            return []
        
        semantic_blocks = []
        model = self.models["general"]
        
        # 获取所有段落文本
        paragraph_texts = [paragraphs[idx] for idx in indices]
        
        # 计算全局相似度矩阵
        embeddings = model.encode(paragraph_texts, convert_to_tensor=True)
        similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        
        # 识别问题段落（主题边界）
        question_indices = []
        for i, text in enumerate(paragraph_texts):
            if self._is_potential_question(text):
                question_indices.append(i)
        
        # 如果没有问题段落，尝试基于相似度进行分组
        if not question_indices:
            return self._similarity_based_grouping(paragraphs, indices, similarity_matrix)
        
        # 基于问题段落进行主题分组
        for i, q_idx in enumerate(question_indices):
            # 确定当前主题的结束位置
            if i + 1 < len(question_indices):
                end_idx = question_indices[i + 1]
            else:
                end_idx = len(indices)
            
            # 获取当前主题范围内的段落
            topic_start = q_idx
            topic_end = end_idx
            
            # 使用相似度矩阵进行主题内分组
            current_group = [paragraph_texts[q_idx]]  # 从问题开始
            current_indices = [indices[q_idx]]
            
            # 计算动态阈值
            topic_paragraphs = paragraph_texts[topic_start:topic_end]
            threshold = self._calculate_dynamic_threshold(topic_paragraphs, model)
            
            # 向前扩展：添加与问题段落相似的段落
            for j in range(q_idx + 1, topic_end):
                # 计算与问题段落的相似度
                question_similarity = similarity_matrix[q_idx, j]
                
                # 计算与组内所有段落的平均相似度
                group_similarities = [similarity_matrix[j, k] for k in range(len(current_group))]
                avg_group_similarity = np.mean(group_similarities)
                
                # 计算与前一段落的相似度（连续性检查）
                prev_similarity = similarity_matrix[j, j-1] if j > 0 else 0
                
                # 更宽松的加入条件：
                # 1. 与问题段落相似度高
                # 2. 与组内段落平均相似度高
                # 3. 与前一段落相似度高（连续性）
                # 4. 或者是下一个问题段落之前的最后一个段落
                should_add = (
                    question_similarity > threshold * 0.7 or  # 降低阈值
                    avg_group_similarity > threshold * 0.8 or  # 降低阈值
                    prev_similarity > threshold * 0.5 or  # 连续性检查
                    j == topic_end - 1  # 主题范围内的最后一个段落
                )
                
                if should_add:
                    current_group.append(paragraph_texts[j])
                    current_indices.append(indices[j])
                else:
                    # 如果相似度都不够，但这是主题范围内的段落，也要加入
                    # 避免在主题中间断开
                    if j < topic_end - 1:  # 不是主题的最后一个段落
                        current_group.append(paragraph_texts[j])
                        current_indices.append(indices[j])
                    else:
                        break
            
            # 创建语义块（降低最小块大小要求，确保所有主题都被处理）
            min_size = max(self.min_block_size // 2, 20)  # 降低最小块大小要求
            if len("\n\n".join(current_group)) >= min_size:
                semantic_blocks.append({
                    'content': "\n\n".join(current_group),
                    'confidence': 'high',
                    'type': 'enhanced_hierarchical',
                    'domain': self._detect_domain("\n\n".join(current_group)),
                    'indices': current_indices
                })
            else:
                # 如果块太小，但仍然包含问题段落，也要保留
                if any(self._is_potential_question(text) for text in current_group):
                    semantic_blocks.append({
                        'content': "\n\n".join(current_group),
                        'confidence': 'medium',
                        'type': 'small_question_block',
                        'domain': self._detect_domain("\n\n".join(current_group)),
                        'indices': current_indices
                    })
        
        self.logger.info(f"Enhanced hierarchical grouping: {len(semantic_blocks)} semantic blocks created")
        return semantic_blocks
    
    def _similarity_based_grouping(self, paragraphs: List[str], indices: List[int], similarity_matrix: np.ndarray) -> List[Dict]:
        """
        基于相似度矩阵的分组：当没有明确问题段落时使用
        """
        if not indices:
            return []
        
        semantic_blocks = []
        paragraph_texts = [paragraphs[idx] for idx in indices]
        
        # 计算动态阈值
        model = self.models["general"]
        threshold = self._calculate_dynamic_threshold(paragraph_texts, model)
        
        # 使用层次聚类思想
        i = 0
        while i < len(indices):
            current_group = [paragraph_texts[i]]
            current_indices = [indices[i]]
            
            # 向前扩展：寻找相似段落
            j = i + 1
            while j < len(indices):
                # 计算与组内所有段落的平均相似度
                group_similarities = [similarity_matrix[j, k] for k in range(i, j)]
                avg_similarity = np.mean(group_similarities) if group_similarities else 0
                
                if avg_similarity > threshold:
                    current_group.append(paragraph_texts[j])
                    current_indices.append(indices[j])
                    j += 1
                else:
                    break
            
            # 创建语义块
            if len("\n\n".join(current_group)) >= self.min_block_size:
                semantic_blocks.append({
                    'content': "\n\n".join(current_group),
                    'confidence': 'medium',
                    'type': 'similarity_based',
                    'domain': self._detect_domain("\n\n".join(current_group)),
                    'indices': current_indices
                })
            
            i = j
        
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
        🔥 增强版本：添加详细的调试信息和处理报告
        """
        if not paragraphs:
            return []
        
        self.logger.info(f"🚀 Starting semantic grouping for {len(paragraphs)} paragraphs")
        
        # 🔥 调试信息：输出原始段落信息
        self._log_paragraph_analysis(paragraphs)
        
        # 第一层：规则预筛选
        high_confidence_blocks, remaining_indices = self._rule_based_prescreening(paragraphs)
        
        # 第二层：改进的语义分组（处理剩余段落）
        semantic_blocks = []
        if remaining_indices:
            # 优先使用层次聚类分组，如果失败则回退到改进分组
            try:
                semantic_blocks = self._hierarchical_semantic_grouping(paragraphs, remaining_indices)
                if not semantic_blocks:  # 如果层次分组没有结果，使用改进分组
                    semantic_blocks = self._improved_semantic_grouping(paragraphs, remaining_indices)
            except Exception as e:
                self.logger.warning(f"Hierarchical grouping failed, falling back to improved grouping: {e}")
                semantic_blocks = self._improved_semantic_grouping(paragraphs, remaining_indices)
        
        # 合并所有块
        all_blocks = high_confidence_blocks + semantic_blocks
        
        # 第三层：优化和合并
        optimized_blocks = self._merge_and_optimize_blocks(all_blocks)
        
        # 🔥 调试信息：输出最终统计报告
        self._log_grouping_summary(paragraphs, optimized_blocks, remaining_indices)
        
        return optimized_blocks
    
    def _log_paragraph_analysis(self, paragraphs: List[str]) -> None:
        """
        🔥 输出段落分析的详细信息
        """
        self.logger.info("📊 Paragraph Analysis:")
        self.logger.info(f"  Total paragraphs: {len(paragraphs)}")
        
        # 分析段落类型
        question_count = 0
        answer_count = 0
        other_count = 0
        
        for i, para in enumerate(paragraphs):
            if self._is_flexible_question(para):
                question_count += 1
                self.logger.debug(f"  P{i+1:02d} [QUESTION]: {para[:80]}...")
            elif self._is_flexible_answer(para):
                answer_count += 1
                self.logger.debug(f"  P{i+1:02d} [ANSWER  ]: {para[:80]}...")
            else:
                other_count += 1
                self.logger.debug(f"  P{i+1:02d} [OTHER   ]: {para[:80]}...")
        
        self.logger.info(f"  - Questions detected: {question_count}")
        self.logger.info(f"  - Answers detected: {answer_count}")
        self.logger.info(f"  - Other paragraphs: {other_count}")
        
        # 长度分析
        lengths = [len(para) for para in paragraphs]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        self.logger.info(f"  - Average paragraph length: {avg_length:.1f} chars")
        self.logger.info(f"  - Length range: {min(lengths)}-{max(lengths)} chars")
    
    def _log_grouping_summary(self, original_paragraphs: List[str], final_blocks: List[Dict], remaining_indices: List[int]) -> None:
        """
        🔥 输出分组过程的详细统计报告
        """
        self.logger.info("📈 Semantic Grouping Summary:")
        self.logger.info("=" * 50)
        
        # 基本统计
        total_chars = sum(len(para) for para in original_paragraphs)
        processed_chars = sum(len(block['content']) for block in final_blocks)
        
        self.logger.info(f"📊 Processing Statistics:")
        self.logger.info(f"  - Input paragraphs: {len(original_paragraphs)}")
        self.logger.info(f"  - Output blocks: {len(final_blocks)}")
        self.logger.info(f"  - Total input chars: {total_chars:,}")
        self.logger.info(f"  - Total processed chars: {processed_chars:,}")
        self.logger.info(f"  - Processing coverage: {(processed_chars/total_chars)*100:.1f}%")
        
        # 置信度分布
        confidence_stats = {}
        type_stats = {}
        
        for block in final_blocks:
            conf = block.get('confidence', 'unknown')
            type_name = block.get('type', 'unknown')
            
            confidence_stats[conf] = confidence_stats.get(conf, 0) + 1
            type_stats[type_name] = type_stats.get(type_name, 0) + 1
        
        self.logger.info(f"🎯 Confidence Distribution:")
        for conf, count in sorted(confidence_stats.items()):
            percentage = (count / len(final_blocks)) * 100 if final_blocks else 0
            self.logger.info(f"  - {conf.title()}: {count} blocks ({percentage:.1f}%)")
        
        self.logger.info(f"🔧 Block Types:")
        for type_name, count in sorted(type_stats.items()):
            percentage = (count / len(final_blocks)) * 100 if final_blocks else 0
            self.logger.info(f"  - {type_name}: {count} blocks ({percentage:.1f}%)")
        
        # 块大小分析
        if final_blocks:
            block_sizes = [len(block['content']) for block in final_blocks]
            avg_size = sum(block_sizes) / len(block_sizes)
            
            self.logger.info(f"📏 Block Size Analysis:")
            self.logger.info(f"  - Average size: {avg_size:.1f} chars")
            self.logger.info(f"  - Size range: {min(block_sizes)}-{max(block_sizes)} chars")
            self.logger.info(f"  - Blocks under {self.min_block_size}: {sum(1 for s in block_sizes if s < self.min_block_size)}")
            self.logger.info(f"  - Blocks over {self.max_block_size}: {sum(1 for s in block_sizes if s > self.max_block_size)}")
        
        # 🔥 详细的块信息（如果启用DEBUG级别）
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("🔍 Detailed Block Information:")
            for i, block in enumerate(final_blocks):
                size = len(block['content'])
                conf = block.get('confidence', 'unknown')
                type_name = block.get('type', 'unknown')
                self.logger.debug(f"  Block {i+1:02d}: {size:4d} chars, {conf:7s}, {type_name}")
                self.logger.debug(f"    Content preview: {block['content'][:100]}...")
        
        self.logger.info("=" * 50) 