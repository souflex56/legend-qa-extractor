#!/usr/bin/env python3
"""
Excel到Golden Set转换工具
支持从Excel表格快速生成evaluation所需的golden_set.jsonl文件

依赖安装：
pip install pandas openpyxl

使用方法：
1. 填写Excel模板 (golden_set_template.xlsx)
2. 运行: python scripts/excel_to_golden_set.py
3. 生成: golden_set.jsonl

Excel表格列名要求：
- question (必需): 问题
- answer (必需): 答案  
- source_text (可选): 原始文本
- domain (可选): 领域分类
- difficulty (可选): 难度等级
- answer_type (可选): 答案类型
- quality_score (可选): 质量评分
- notes (可选): 备注说明
"""

import pandas as pd
import json
import os
import sys
from typing import Dict, Any, List

class ExcelToGoldenSetConverter:
    """Excel到Golden Set转换器"""
    
    def __init__(self, 
                 excel_path: str = "golden_set_template.xlsx",
                 output_path: str = "golden_set.jsonl"):
        """
        初始化转换器
        
        Args:
            excel_path: Excel文件路径
            output_path: 输出JSONL文件路径
        """
        self.excel_path = excel_path
        self.output_path = output_path
        
        # 必需字段
        self.required_fields = ["question", "answer"]
        
        # 可选字段及其默认值
        self.optional_fields = {
            "source_text": "",
            "domain": "general",
            "difficulty": "medium", 
            "answer_type": "direct",
            "quality_score": 5,
            "notes": ""
        }
    
    def create_excel_template(self) -> None:
        """创建Excel模板文件"""
        print(f"📋 创建Excel模板: {self.excel_path}")
        
        # 示例数据
        sample_data = [
            {
                "question": "什么是stop doing list？",
                "answer": "所谓要做对的事情实际上是通过不做不对的事情来实现的。",
                "source_text": "网友：什么是stop doing list？\n\n段永平：所谓要做对的事情实际上是通过不做不对的事情来实现的。",
                "domain": "business",
                "difficulty": "medium",
                "answer_type": "explanatory", 
                "quality_score": 5,
                "notes": "经典的stop doing list理念"
            },
            {
                "question": "价值投资的核心是什么？",
                "answer": "价值投资的核心就是买股票就是买公司，买公司就是买这家公司的生意。",
                "source_text": "网友：价值投资的核心是什么？\n\n段永平：价值投资的核心就是买股票就是买公司，买公司就是买这家公司的生意。",
                "domain": "investment",
                "difficulty": "easy",
                "answer_type": "direct",
                "quality_score": 5,
                "notes": "巴菲特投资理念的核心表述"
            },
            {
                "question": "如何看待市场波动？",
                "answer": "市场先生的报价每天都不一样，你要把他当成一个躁郁症患者，不要太把他的话当真。",
                "source_text": "网友：如何看待市场波动？\n\n段永平：市场先生的报价每天都不一样，你要把他当成一个躁郁症患者，不要太把他的话当真。",
                "domain": "investment", 
                "difficulty": "medium",
                "answer_type": "opinion",
                "quality_score": 4,
                "notes": "引用巴菲特的市场先生理论"
            }
        ]
        
        # 创建DataFrame
        df = pd.DataFrame(sample_data)
        
        # 保存到Excel
        with pd.ExcelWriter(self.excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='GoldenSet', index=False)
            
            # 获取工作表
            worksheet = writer.sheets['GoldenSet']
            
            # 自动调整列宽
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)  # 最大宽度50
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"✅ Excel模板已创建: {self.excel_path}")
        print("📝 请编辑Excel文件后运行转换功能")
        
        # 打印字段说明
        print("\n📋 Excel表格字段说明:")
        print("  必需字段:")
        print("    • question: 问题文本")
        print("    • answer: 答案文本") 
        print("  可选字段:")
        print("    • source_text: 原始来源文本")
        print("    • domain: 领域分类 (general/investment/business等)")
        print("    • difficulty: 难度等级 (easy/medium/hard)")
        print("    • answer_type: 答案类型 (direct/explanatory/opinion等)")
        print("    • quality_score: 质量评分 (1-5分)")
        print("    • notes: 备注说明")
    
    def validate_excel_data(self, df: pd.DataFrame) -> List[str]:
        """验证Excel数据格式"""
        errors = []
        
        # 检查必需字段
        for field in self.required_fields:
            if field not in df.columns:
                errors.append(f"缺少必需字段: {field}")
            elif df[field].isnull().any():
                null_rows = df[df[field].isnull()].index.tolist()
                errors.append(f"字段 {field} 在第 {[r+2 for r in null_rows]} 行为空")  # +2因为Excel从第2行开始
        
        # 检查数据类型
        for idx, row in df.iterrows():
            if 'question' in df.columns and pd.notna(row['question']):
                if not isinstance(row['question'], str) or len(str(row['question']).strip()) == 0:
                    errors.append(f"第 {idx+2} 行的问题格式无效")
            
            if 'answer' in df.columns and pd.notna(row['answer']):
                if not isinstance(row['answer'], str) or len(str(row['answer']).strip()) == 0:
                    errors.append(f"第 {idx+2} 行的答案格式无效")
            
            if 'quality_score' in df.columns and pd.notna(row['quality_score']):
                try:
                    score = float(row['quality_score'])
                    if score < 1 or score > 5:
                        errors.append(f"第 {idx+2} 行的质量评分应在1-5之间")
                except (ValueError, TypeError):
                    errors.append(f"第 {idx+2} 行的质量评分格式无效")
        
        return errors
    
    def convert_excel_to_jsonl(self) -> None:
        """转换Excel文件到JSONL格式"""
        if not os.path.exists(self.excel_path):
            print(f"❌ Excel文件不存在: {self.excel_path}")
            print("正在创建模板文件...")
            self.create_excel_template()
            return
        
        print(f"📖 读取Excel文件: {self.excel_path}")
        
        try:
            # 读取Excel文件
            df = pd.read_excel(self.excel_path, sheet_name=0)  # 读取第一个工作表
            
            print(f"📊 读取到 {len(df)} 行数据")
            
            # 验证数据
            errors = self.validate_excel_data(df)
            if errors:
                print("❌ 数据验证失败:")
                for error in errors:
                    print(f"   • {error}")
                return
            
            # 转换数据
            qa_pairs = []
            skipped_rows = 0
            
            for idx, row in df.iterrows():
                try:
                    qa_pair = {}
                    
                    # 处理必需字段
                    for field in self.required_fields:
                        if field in df.columns and pd.notna(row[field]):
                            qa_pair[field] = str(row[field]).strip()
                        else:
                            print(f"⚠️  第 {idx+2} 行缺少必需字段 {field}，跳过")
                            skipped_rows += 1
                            break
                    else:
                        # 处理可选字段
                        for field, default_value in self.optional_fields.items():
                            if field in df.columns and pd.notna(row[field]):
                                value = row[field]
                                # 特殊处理数值字段
                                if field == 'quality_score':
                                    try:
                                        qa_pair[field] = float(value)
                                    except (ValueError, TypeError):
                                        qa_pair[field] = default_value
                                else:
                                    qa_pair[field] = str(value).strip() if str(value).strip() else default_value
                            # 只在非空值时添加可选字段
                            elif field in df.columns or field in ['domain', 'quality_score']:  # 保留重要的可选字段
                                qa_pair[field] = default_value
                        
                        qa_pairs.append(qa_pair)
                
                except Exception as e:
                    print(f"⚠️  第 {idx+2} 行处理失败: {e}")
                    skipped_rows += 1
            
            if not qa_pairs:
                print("❌ 没有有效的QA对可以转换")
                return
            
            # 保存为JSONL文件
            print(f"💾 保存到文件: {self.output_path}")
            
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for qa_pair in qa_pairs:
                    f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
            
            print(f"✅ 转换完成!")
            print(f"   • 成功转换: {len(qa_pairs)} 个QA对")
            if skipped_rows > 0:
                print(f"   • 跳过行数: {skipped_rows}")
            print(f"   • 输出文件: {self.output_path}")
            
            # 显示转换后的前几个样例
            print(f"\n📋 转换样例 (前3个):")
            for i, qa in enumerate(qa_pairs[:3]):
                print(f"   {i+1}. Q: {qa['question'][:50]}{'...' if len(qa['question']) > 50 else ''}")
                print(f"      A: {qa['answer'][:50]}{'...' if len(qa['answer']) > 50 else ''}")
                print()
        
        except Exception as e:
            print(f"❌ 转换失败: {e}")
            print("请检查Excel文件格式是否正确")

def main():
    """主函数"""
    converter = ExcelToGoldenSetConverter()
    
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        if action == "template":
            converter.create_excel_template()
        elif action == "convert":
            converter.convert_excel_to_jsonl()
        else:
            print("用法: python excel_to_golden_set.py [template|convert]")
    else:
        # 默认行为：如果Excel文件不存在则创建模板，否则执行转换
        if not os.path.exists(converter.excel_path):
            converter.create_excel_template()
        else:
            converter.convert_excel_to_jsonl()

if __name__ == "__main__":
    main() 