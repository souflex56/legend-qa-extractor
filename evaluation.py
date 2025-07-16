#!/usr/bin/env python3
"""
评估脚本 - 第二阶段：建立量化评估体系
用于客观评估 legend-qa-extractor 系统的表现


使用方法：
1. 先创建黄金标准集 golden_set.jsonl（人工标注）可以由excel转换而来 也可以直接编辑jsonl文件
具体参考docs/README_excel_converter.md 创建黄金标准集 
2. 运行此脚本进行评估：python evaluation.py
"""

import json
import os
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

# 设置文件路径
golden_set_path = "golden_set.jsonl"
generated_qa_path = "output/output_final_qa_chunkwise_0.02.jsonl"
evaluation_report_path = "evaluation_report.json"


try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"依赖库缺失，请安装：pip install sentence-transformers scikit-learn")
    print(f"错误详情：{e}")
    exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class QAEvaluator:
    """QA提取系统评估器"""
    
    def __init__(self, 
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 golden_set_path: str = golden_set_path,
                 generated_qa_path: str = generated_qa_path,
                 evaluation_report_path: str = evaluation_report_path):
        """
        初始化评估器
        
        Args:
            model_name: 句子嵌入模型名称
            golden_set_path: 黄金标准集文件路径
            generated_qa_path: 生成的QA文件路径
            evaluation_report_path: 评估报告输出文件路径
        """
        self.golden_set_path = golden_set_path
        self.generated_qa_path = generated_qa_path
        self.evaluation_report_path = evaluation_report_path
        
        logger.info(f"加载句子嵌入模型: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        self.golden_qa_pairs = []
        self.generated_qa_pairs = []
    
    def load_qa_pairs(self, file_path: str) -> List[Dict[str, Any]]:
        """加载QA对数据"""
        qa_pairs = []
        
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return qa_pairs
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        qa_pair = json.loads(line.strip())
                        if self._is_valid_qa_pair(qa_pair):
                            qa_pairs.append(qa_pair)
                        else:
                            logger.warning(f"第{line_num}行QA对格式无效: {line.strip()[:100]}...")
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
            
            logger.info(f"从 {file_path} 加载了 {len(qa_pairs)} 个有效QA对")
            
        except Exception as e:
            logger.error(f"加载文件 {file_path} 时发生错误: {e}")
        
        return qa_pairs
    
    def _is_valid_qa_pair(self, qa_pair: Dict[str, Any]) -> bool:
        """验证QA对格式是否有效"""
        return (isinstance(qa_pair, dict) and 
                "question" in qa_pair and "answer" in qa_pair and
                qa_pair.get("question") and qa_pair.get("answer") and
                isinstance(qa_pair["question"], str) and 
                isinstance(qa_pair["answer"], str))
    
    def find_most_similar_question(self, target_question: str, candidate_qa_pairs: List[Dict[str, Any]]) -> Tuple[int, float]:
        """
        在候选QA对中找到最相似的问题
        
        Args:
            target_question: 目标问题
            candidate_qa_pairs: 候选QA对列表
            
        Returns:
            (best_match_index, similarity_score)
        """
        if not candidate_qa_pairs:
            return -1, 0.0
        
        candidate_questions = [qa["question"] for qa in candidate_qa_pairs]
        
        # 计算嵌入向量
        target_embedding = self.embedding_model.encode([target_question])
        candidate_embeddings = self.embedding_model.encode(candidate_questions)
        
        # 计算余弦相似度
        similarities = cosine_similarity(target_embedding, candidate_embeddings)[0]
        
        # 找到最高相似度
        best_index = np.argmax(similarities)
        best_similarity = similarities[best_index]
        
        return best_index, best_similarity
    
    def calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
        """计算两个答案之间的相似度"""
        embeddings = self.embedding_model.encode([answer1, answer2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def evaluate_qa_extraction(self) -> Dict[str, Any]:
        """
        主评估函数：对比生成的QA对与黄金标准集
        
        Returns:
            评估结果字典
        """
        logger.info("开始QA提取质量评估...")
        
        # 加载数据
        self.golden_qa_pairs = self.load_qa_pairs(self.golden_set_path)
        self.generated_qa_pairs = self.load_qa_pairs(self.generated_qa_path)
        
        if not self.golden_qa_pairs:
            logger.error("黄金标准集为空，无法进行评估")
            return {"error": "黄金标准集为空"}
        
        if not self.generated_qa_pairs:
            logger.error("生成的QA对为空，无法进行评估")
            return {"error": "生成的QA对为空"}
        
        # 评估指标
        evaluation_results = {
            "golden_set_size": len(self.golden_qa_pairs),
            "generated_set_size": len(self.generated_qa_pairs),
            "question_similarities": [],
            "answer_similarities": [],
            "matched_pairs": 0,
            "average_question_similarity": 0.0,
            "average_answer_similarity": 0.0,
            "overall_score": 0.0,
            "detailed_matches": []
        }
        
        logger.info(f"黄金标准集大小: {len(self.golden_qa_pairs)}")
        logger.info(f"生成集大小: {len(self.generated_qa_pairs)}")
        
        # 对每个黄金标准QA对，在生成集中找最相似的
        question_similarities = []
        answer_similarities = []
        
        for i, golden_qa in enumerate(self.golden_qa_pairs):
            golden_question = golden_qa["question"]
            golden_answer = golden_qa["answer"]
            
            # 在生成集中找最相似的问题
            best_match_idx, question_sim = self.find_most_similar_question(
                golden_question, self.generated_qa_pairs
            )
            
            if best_match_idx >= 0:
                matched_generated_qa = self.generated_qa_pairs[best_match_idx]
                generated_answer = matched_generated_qa["answer"]
                
                # 计算答案相似度
                answer_sim = self.calculate_answer_similarity(golden_answer, generated_answer)
                
                question_similarities.append(question_sim)
                answer_similarities.append(answer_sim)
                
                # 存储详细匹配信息
                evaluation_results["detailed_matches"].append({
                    "golden_index": int(i),
                    "matched_generated_index": int(best_match_idx),
                    "question_similarity": float(question_sim),
                    "answer_similarity": float(answer_sim),
                    "golden_question": golden_question[:100] + "..." if len(golden_question) > 100 else golden_question,
                    "matched_question": matched_generated_qa["question"][:100] + "..." if len(matched_generated_qa["question"]) > 100 else matched_generated_qa["question"],
                    "golden_answer": golden_answer[:100] + "..." if len(golden_answer) > 100 else golden_answer,
                    "matched_answer": generated_answer[:100] + "..." if len(generated_answer) > 100 else generated_answer
                })
                
                logger.debug(f"黄金标准 {i+1}: 问题相似度={question_sim:.3f}, 答案相似度={answer_sim:.3f}")
        
        # 计算统计指标
        if question_similarities:
            evaluation_results["question_similarities"] = [float(x) for x in question_similarities]
            evaluation_results["answer_similarities"] = [float(x) for x in answer_similarities]
            evaluation_results["matched_pairs"] = len(question_similarities)
            evaluation_results["average_question_similarity"] = float(np.mean(question_similarities))
            evaluation_results["average_answer_similarity"] = float(np.mean(answer_similarities))
            
            # 综合分数：问题和答案相似度的加权平均
            evaluation_results["overall_score"] = float(
                0.3 * evaluation_results["average_question_similarity"] + 
                0.7 * evaluation_results["average_answer_similarity"]
            )
            
            # 🚀 新增：等级评定
            evaluation_results["grade"] = self._calculate_grade(evaluation_results["overall_score"])
            
            # 🚀 新增：置信度分布统计
            evaluation_results["confidence_distribution"] = self._analyze_confidence_distribution()
        
        return evaluation_results
    
    def _calculate_grade(self, overall_score: float) -> str:
        """
        根据综合评分计算等级
        
        Args:
            overall_score: 综合评分 (0.0-1.0)
            
        Returns:
            等级字符串
        """
        if overall_score >= 0.9:
            return "优秀 🏆"
        elif overall_score >= 0.8:
            return "良好 👍"
        elif overall_score >= 0.7:
            return "一般 👌"
        elif overall_score >= 0.6:
            return "及格 ⚠️"
        else:
            return "需要改进 ❌"
    
    def _analyze_confidence_distribution(self) -> Dict[str, str]:
        """
        分析生成集的置信度分布
        
        Returns:
            置信度分布统计字典
        """
        confidence_stats = {}
        total_count = len(self.generated_qa_pairs)
        
        if total_count == 0:
            return {"error": "No generated QA pairs to analyze"}
        
        # 统计各置信度级别的数量
        confidence_counts = {}
        for qa_pair in self.generated_qa_pairs:
            confidence = qa_pair.get('source_confidence', 'unknown')
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        
        # 转换为百分比格式
        for confidence, count in confidence_counts.items():
            percentage = (count / total_count) * 100
            confidence_stats[confidence] = f"{count} blocks ({percentage:.1f}%)"
        
        return confidence_stats
    
    def generate_evaluation_report(self, results: Dict[str, Any], output_path: str = None):
        """生成详细的评估报告"""
        
        # 如果没有指定输出路径，使用默认设置
        if output_path is None:
            output_path = self.evaluation_report_path
        
        # 保存完整结果到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 打印摘要报告
        print("\n" + "="*80)
        print("QA提取系统评估报告")
        print("="*80)
        
        if "error" in results:
            print(f"❌ 评估失败: {results['error']}")
            return
        
        print(f"📊 数据统计:")
        print(f"   黄金标准集大小: {results['golden_set_size']}")
        print(f"   生成集大小: {results['generated_set_size']}")
        print(f"   成功匹配对数: {results['matched_pairs']}")
        
        print(f"\n📈 相似度指标:")
        print(f"   平均问题相似度: {results['average_question_similarity']:.4f}")
        print(f"   平均答案相似度: {results['average_answer_similarity']:.4f}")
        print(f"   综合评分: {results['overall_score']:.4f}")
        print(f"   系统评级: {results.get('grade', '未知')}")
        
        # 🚀 新增：显示置信度分布
        confidence_dist = results.get('confidence_distribution', {})
        if confidence_dist and 'error' not in confidence_dist:
            print(f"\n🎯 置信度分布:")
            for confidence, stats in confidence_dist.items():
                confidence_name = {
                    'high': '高置信度',
                    'medium': '中置信度', 
                    'low': '低置信度',
                    'unknown': '未知'
                }.get(confidence, confidence)
                print(f"   {confidence_name}: {stats}")
        
        print(f"\n📁 详细报告已保存至: {output_path}")
        print("="*80)
    
    def create_sample_golden_set(self, sample_path: str = "golden_set_sample.jsonl"):
        """创建示例黄金标准集文件（供参考）"""
        sample_data = [
            {
                "question": "什么是stop doing list？",
                "answer": "所谓要做对的事情实际上是通过不做不对的事情来实现的。"
            },
            {
                "question": "价值投资的核心是什么？",
                "answer": "价值投资的核心就是买股票就是买公司，买公司就是买这家公司的生意。"
            },
            {
                "question": "如何看待市场波动？",
                "answer": "市场先生的报价每天都不一样，你要把他当成一个躁郁症患者，不要太把他的话当真。"
            }
        ]
        
        with open(sample_path, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"示例黄金标准集已创建: {sample_path}")
        print(f"📝 示例黄金标准集已创建: {sample_path}")
        print("请根据您的实际需求，创建包含30-50个高质量QA对的 golden_set.jsonl 文件")


def main():
    """主函数"""
    evaluator = QAEvaluator(
        golden_set_path=golden_set_path,
        generated_qa_path=generated_qa_path,
        evaluation_report_path=evaluation_report_path
    )
    
    # 检查是否存在黄金标准集
    if not os.path.exists(evaluator.golden_set_path):
        print(f"❌ 黄金标准集文件不存在: {evaluator.golden_set_path}")
        print("正在创建示例文件...")
        evaluator.create_sample_golden_set()
        return
    
    # 检查是否存在生成的QA文件
    if not os.path.exists(evaluator.generated_qa_path):
        print(f"❌ 生成的QA文件不存在: {evaluator.generated_qa_path}")
        print("请先运行主流程生成QA文件")
        return
    
    # 执行评估
    results = evaluator.evaluate_qa_extraction()
    
    # 生成报告
    evaluator.generate_evaluation_report(results)


if __name__ == "__main__":
    main() 