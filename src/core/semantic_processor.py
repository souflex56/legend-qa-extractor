
import re
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .text_processor import TextProcessor

logger = logging.getLogger(__name__)

class SemanticProcessor:
    """
    Handles semantic text grouping using a hybrid approach of rules and embedding similarity.
    This processor replaces the traditional "chunking" with "semantic grouping".
    """
    def __init__(self, 
                 text_processor: TextProcessor,
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 device: Optional[str] = None):
        """
        Initializes the SemanticProcessor.

        Args:
            text_processor: An instance of TextProcessor for text cleaning and pattern matching.
            model_name: The name of the sentence-transformer model to use.
            device: The device to run the model on (e.g., 'cpu', 'cuda').
        """
        self.text_processor = text_processor
        self.logger = logger
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.logger.info(f"✅ SentenceTransformer model '{model_name}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"🔥 Failed to load SentenceTransformer model '{model_name}'. Please ensure it is installed and accessible.")
            self.logger.error(f"You may need to run: pip install -U sentence-transformers")
            self.model = None
            raise e

    def group_text_by_semantics(self, text: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main entry point for processing a document. It groups text based on semantic boundaries
        and filtering level specified in the config.

        Args:
            text: The raw text of the document.
            config: A dictionary containing processing parameters like 'filtering_level', 
                    'semantic_threshold', 'max_question_length', etc.

        Returns:
            A list of dictionaries, where each dictionary represents a semantic group
            ready for processing by the LLM.
        """
        if not self.model:
            self.logger.error("Semantic model is not available. Aborting processing.")
            return []

        if not text:
            return []

        # 1. Split text into paragraphs, the basic building blocks
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # 2. First pass: Identify high-confidence groups using regex
        rule_based_groups, remaining_indices = self._group_by_rules(paragraphs)
        
        # 3. Handle different filtering levels
        filtering_level = config.get('filtering_level', 'balanced')
        
        if filtering_level == 'strict':
            self.logger.info(f"Filtering level is 'strict'. Returning {len(rule_based_groups)} high-confidence groups.")
            return [{"content": group, "type": "high-confidence"} for group in rule_based_groups]

        # For 'balanced' and 'none', we need to process the remaining paragraphs
        remaining_paragraphs = [paragraphs[i] for i in remaining_indices]
        
        # 4. Second pass: Identify medium-confidence groups using semantic similarity
        semantic_groups, ungrouped_paras = self._group_by_semantics(
            remaining_paragraphs, 
            config.get('max_question_length', 50),
            config.get('semantic_threshold', 0.5)
        )

        # 5. Combine and finalize groups
        all_groups = []
        all_groups.extend([{"content": group, "type": "high-confidence"} for group in rule_based_groups])
        all_groups.extend([{"content": group, "type": "medium-confidence"} for group in semantic_groups])
        
        if filtering_level == 'none':
            # Add all remaining paragraphs as low-confidence groups
            all_groups.extend([{"content": para, "type": "low-confidence"} for para in ungrouped_paras])
        
        self.logger.info(f"Total groups created: {len(all_groups)} (High: {len(rule_based_groups)}, Medium: {len(semantic_groups)}, Low: {len(ungrouped_paras) if filtering_level == 'none' else 0})")
        return all_groups
    
    def _group_by_rules(self, paragraphs: List[str]) -> (List[str], List[int]):
        """
        Groups paragraphs based on high-confidence Q&A start patterns.
        A group starts with a question pattern and ends before the next question pattern.
        This implementation is now more conservative to avoid overly greedy grouping.
        """
        groups = []
        used_indices = set()
        
        i = 0
        while i < len(paragraphs):
            if i in used_indices:
                i += 1
                continue

            # Check if the current paragraph is a potential start of a Q&A
            if self.text_processor.block_has_qa(paragraphs[i]):
                start_index = i
                
                # More conservative rule: Group a question with its immediate next paragraph
                # if that paragraph does NOT look like another question.
                if i + 1 < len(paragraphs) and not self.text_processor.block_has_qa(paragraphs[i + 1]):
                    # This is likely a Question-Answer pair.
                    end_index = i + 2
                    group_paragraphs = paragraphs[start_index:end_index]
                    groups.append("\n\n".join(group_paragraphs))
                    used_indices.add(i)
                    used_indices.add(i + 1)
                    i += 2
                else:
                    # This is either a standalone block (e.g., contains Q&A within itself)
                    # or a question followed by another question. Treat it as a single-para group.
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

    def _group_by_semantics(self, paragraphs: List[str], max_question_len: int, threshold: float) -> (List[str], List[str]):
        """
        Groups remaining paragraphs using semantic similarity.
        It finds short "potential questions" and groups them with subsequent, semantically
        related paragraphs.
        """
        if not paragraphs:
            return [], []
            
        groups = []
        used_indices = set()
        
        # 1. Identify potential questions
        potential_questions = {}
        for i, p in enumerate(paragraphs):
            if len(p) <= max_question_len:
                # A simple heuristic: could be a question. More advanced logic could be added here.
                potential_questions[i] = p
        
        if not potential_questions:
            return [], paragraphs

        # 2. Encode all paragraphs for semantic comparison
        embeddings = self.model.encode(paragraphs, convert_to_tensor=True)

        # 3. Iterate through potential questions and find related answers
        for q_index, q_text in potential_questions.items():
            if q_index in used_indices:
                continue

            current_group_indices = {q_index}
            q_embedding = embeddings[q_index].cpu().numpy().reshape(1, -1)

            # Look ahead in subsequent paragraphs
            for a_index in range(q_index + 1, len(paragraphs)):
                if a_index in used_indices:
                    # If we hit a paragraph that's already part of another group, stop.
                    break
                
                a_embedding = embeddings[a_index].cpu().numpy().reshape(1, -1)
                
                # Calculate similarity between the question and the potential answer
                similarity = cosine_similarity(q_embedding, a_embedding)[0][0]
                
                if similarity >= threshold:
                    # This paragraph is semantically related, add it to the group
                    current_group_indices.add(a_index)
                else:
                    # The semantic link is broken, stop forming this group
                    break
            
            if len(current_group_indices) > 1: # Found a valid group (question + at least one answer)
                # Sort indices to maintain original order
                sorted_indices = sorted(list(current_group_indices))
                group_paras = [paragraphs[i] for i in sorted_indices]
                groups.append("\n\n".join(group_paras))
                used_indices.update(sorted_indices)

        # 4. Collect all paragraphs that were not grouped
        ungrouped_paras = [p for i, p in enumerate(paragraphs) if i not in used_indices]
        
        self.logger.info(f"Identified {len(groups)} medium-confidence groups using semantics.")
        return groups, ungrouped_paras 