# src/core/semantic_grouper.py
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy # 或者 HanLP

class SemanticGrouper:
    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load("zh_core_web_sm") # 用于语法分析
        self.models = {
            "general": SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'),
            # 可以预留位置给领域模型
            # "financial": SentenceTransformer('path/to/finbert'),
        }

    def _detect_domain(self, text_chunk):
        # 简化版领域检测，面试时可扩展讨论
        if "$" in text_chunk or "财报" in text_chunk:
            return "financial"
        return "general"

    def _is_potential_question(self, paragraph: str) -> bool:
        # 结合长度和语法进行判断
        max_length = getattr(self.config, 'max_question_length', 50)
        if len(paragraph) > max_length:
            return False
        doc = self.nlp(paragraph)
        # 判断是否以问号结尾或包含疑问词/句式
        if paragraph.endswith(('?', '？')) or any(token.pos_ == 'ADV' and '疑问' in token.tag_ for token in doc):
            return True
        return False

    def _calculate_dynamic_threshold(self, group_embeddings) -> float:
        """Calculate dynamic similarity threshold based on a group of embeddings."""
        default_threshold = getattr(self.config, 'default_similarity_threshold', 0.65)
        if len(group_embeddings) < 2:
            return default_threshold

        # Calculate cosine similarity within the group
        sim_matrix = util.cos_sim(group_embeddings, group_embeddings).cpu().numpy()
        
        # Get similarities from the upper triangle of the matrix (excluding the diagonal)
        upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
        if len(sim_matrix[upper_triangle_indices]) == 0:
            return default_threshold

        # Calculate mean and standard deviation of similarities
        window_avg = np.mean(sim_matrix[upper_triangle_indices])
        window_std = np.std(sim_matrix[upper_triangle_indices])

        # Calculate dynamic threshold
        std_factor = getattr(self.config, 'std_factor', 0.5)
        threshold = window_avg + std_factor * window_std
        return float(np.clip(threshold, 0.5, 0.9))

    def group(self, paragraphs: list[str]) -> list[dict]:
        """Group paragraphs into larger, coherent blocks."""
        if not paragraphs:
            return []

        # Get block size limits from config
        min_size = self.config.min_block_size
        max_size = self.config.max_block_size
        
        all_blocks = []
        i = 0
        while i < len(paragraphs):
            current_block_paras = []
            current_len = 0
            
            # Create a block by adding paragraphs until max_size is reached
            while i < len(paragraphs):
                para = paragraphs[i]
                
                # If adding the next paragraph exceeds max_size, finalize the block
                if current_len > 0 and current_len + len(para) > max_size:
                    break
                
                current_block_paras.append(para)
                current_len += len(para)
                i += 1
                
                # If the block size is already at max, break to finalize
                if current_len >= max_size:
                    break
            
            # If the block has content and meets min_size, add it
            if current_block_paras:
                block_text = "\n\n".join(current_block_paras)
                if len(block_text) >= min_size:
                    all_blocks.append({"content": block_text, "confidence": 1.0})

        return all_blocks 