#!/usr/bin/env python3
"""
展示剩余段落的详细分组过程
P1, P2, P9, P10被规则过滤后，剩余段落的滑动窗口处理
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

def create_remaining_paragraphs_demo():
    """创建剩余段落分组演示"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    fig.suptitle('Remaining Paragraphs Grouping Process\nAfter Rule-based Filtering', 
                 fontsize=16, fontweight='bold')
    
    # ===== 上半部分: 规则过滤过程 =====
    ax1.set_title('Step 1: Rule-based Filtering Results', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    
    # 原始段落序列
    original_paragraphs = [
        ("P1", "Netizen: What about value investing?", '#FFE6CC', True),
        ("P2", "Duan Yongping: Value investing is long-term...", '#E6F3FF', True),
        ("P3", "Reporter: Any management suggestions?", '#FFE6CC', True),
        ("P4", "Duan: Most important thing is culture...", '#E6F3FF', True),
        ("P5", "Background description without clear QA pattern...", '#F0F0F0', False),
        ("P6", "Netizen A: How about Buffett's investment ideas?", '#FFE6CC', True),
        ("P7", "Dadao: Buffett once said about patience...", '#E6F3FF', True),
        ("P8", "Additional notes about compound interest theory...", '#F0F0F0', False),
        ("P9", "Audience: Can you talk about risk control?", '#FFE6CC', True),
        ("P10", "Duan Yongping: Risk control is crucial for...", '#E6F3FF', True)
    ]
    
    # 绘制原始段落
    ax1.text(8, 7.5, "Original Document Paragraphs", ha='center', fontsize=12, fontweight='bold')
    for i, (label, text, color, is_rule_matched) in enumerate(original_paragraphs):
        x_pos = 1 + i * 1.4
        
        # 段落框
        rect = Rectangle((x_pos, 6.5), 1.2, 0.8, facecolor=color, edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(x_pos + 0.6, 6.9, label, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 规则匹配标记
        if is_rule_matched:
            ax1.text(x_pos + 0.6, 6.2, '✓', ha='center', va='center', fontsize=12, 
                    color='green', fontweight='bold')
        else:
            ax1.text(x_pos + 0.6, 6.2, '?', ha='center', va='center', fontsize=12, 
                    color='red', fontweight='bold')
    
    # 规则处理结果
    ax1.text(8, 5.5, "Rule-based Processing Results", ha='center', fontsize=12, fontweight='bold')
    
    # 高置信度块
    rule_blocks = [
        ("Block 1", "P1+P2", 2.4, '#90EE90'),
        ("Block 2", "P3+P4", 5.6, '#90EE90'),  
        ("Block 3", "P6+P7", 9.8, '#90EE90'),
        ("Block 4", "P9+P10", 13.4, '#90EE90')
    ]
    
    for name, paras, x_center, color in rule_blocks:
        rect = FancyBboxPatch((x_center-0.9, 4.2), 1.8, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor='darkgreen',
                             linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x_center, 4.6, f"{name}\n{paras}", ha='center', va='center', 
                fontsize=9, fontweight='bold')
        
        # 连接线
        ax1.plot([x_center, x_center], [5.0, 6.5], 'g--', linewidth=1.5)
    
    # 剩余段落
    remaining_rect = FancyBboxPatch((6.5, 2.5), 3, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#FFCCCC',
                                   edgecolor='red',
                                   linewidth=2)
    ax1.add_patch(remaining_rect)
    ax1.text(8, 3, "Remaining Paragraphs\nP5, P8\nNeed Semantic Processing", 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 连接剩余段落
    ax1.plot([8, 8], [3.5, 6.5], 'r--', linewidth=2)
    ax1.plot([12.2, 12.2], [3.5, 6.5], 'r--', linewidth=2)
    
    # 图例
    ax1.text(1, 1.5, "Legend:", fontsize=10, fontweight='bold')
    ax1.text(1, 1.2, "✓ = Matched by rules", fontsize=9, color='green')
    ax1.text(1, 0.9, "? = Needs semantic processing", fontsize=9, color='red')
    ax1.text(1, 0.6, "Green blocks = High confidence QA pairs", fontsize=9, color='darkgreen')
    
    # ===== 下半部分: 滑动窗口语义处理 =====
    ax2.set_title('Step 2: Sliding Window Semantic Processing for Remaining Paragraphs', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 16)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # 显示完整段落序列（包含已处理的）
    ax2.text(8, 9.5, "Complete Paragraph Sequence with Processing Status", 
            ha='center', fontsize=12, fontweight='bold')
    
    all_paras_status = [
        ("P1", "✓ Processed", '#90EE90'),
        ("P2", "✓ Processed", '#90EE90'),
        ("P3", "✓ Processed", '#90EE90'),
        ("P4", "✓ Processed", '#90EE90'),
        ("P5", "? Remaining", '#FFCCCC'),
        ("P6", "✓ Processed", '#90EE90'),
        ("P7", "✓ Processed", '#90EE90'),
        ("P8", "? Remaining", '#FFCCCC'),
        ("P9", "✓ Processed", '#90EE90'),
        ("P10", "✓ Processed", '#90EE90')
    ]
    
    for i, (para, status, color) in enumerate(all_paras_status):
        x_pos = 1 + i * 1.4
        rect = Rectangle((x_pos, 8.5), 1.2, 0.6, facecolor=color, edgecolor='black', linewidth=1)
        ax2.add_patch(rect)
        ax2.text(x_pos + 0.6, 8.8, para, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 滑动窗口处理演示
    ax2.text(8, 7.8, "Sliding Window Processing (window_size = 5)", 
            ha='center', fontsize=12, fontweight='bold', color='blue')
    
    # 窗口1: P3, P4, P5, P6, P7
    ax2.text(2, 7.2, "Window 1:", fontsize=11, fontweight='bold', color='blue')
    window1_rect = Rectangle((2.6, 7), 7, 0.4, facecolor='none', 
                            edgecolor='blue', linewidth=2, linestyle='--')
    ax2.add_patch(window1_rect)
    ax2.text(6.1, 7.2, "P3, P4, P5, P6, P7", ha='center', fontsize=10, color='blue')
    
    # 窗口1处理逻辑
    ax2.text(1, 6.2, "Window 1 Processing:", fontsize=11, fontweight='bold')
    ax2.text(1, 5.9, "• P3, P4, P6, P7 already processed by rules ✓", fontsize=9)
    ax2.text(1, 5.6, "• P5 needs semantic processing", fontsize=9, color='red')
    ax2.text(1, 5.3, "• Check P5 similarity with neighbors:", fontsize=9)
    ax2.text(1, 5.0, "  - P4 ↔ P5: similarity = 0.42 < threshold(0.55)", fontsize=9, color='red')
    ax2.text(1, 4.7, "  - P5 ↔ P6: similarity = 0.38 < threshold(0.55)", fontsize=9, color='red')
    ax2.text(1, 4.4, "• Result: P5 becomes standalone block", fontsize=9, color='orange')
    
    # 窗口2: P6, P7, P8, P9, P10
    ax2.text(9, 6.2, "Window 2:", fontsize=11, fontweight='bold', color='green')
    window2_rect = Rectangle((8.4, 6), 7, 0.4, facecolor='none', 
                            edgecolor='green', linewidth=2, linestyle='--')
    ax2.add_patch(window2_rect)
    ax2.text(11.9, 6.2, "P6, P7, P8, P9, P10", ha='center', fontsize=10, color='green')
    
    # 窗口2处理逻辑
    ax2.text(9, 5.9, "Window 2 Processing:", fontsize=11, fontweight='bold')
    ax2.text(9, 5.6, "• P6, P7, P9, P10 already processed ✓", fontsize=9)
    ax2.text(9, 5.3, "• P8 needs semantic processing", fontsize=9, color='red')
    ax2.text(9, 5.0, "• Check P8 with consecutive P7:", fontsize=9)
    ax2.text(9, 4.7, "  - P7 ↔ P8: similarity = 0.68 > threshold(0.55)", fontsize=9, color='green')
    ax2.text(9, 4.4, "• Result: Create block P7+P8", fontsize=9, color='green')
    
    # 最终结果
    ax2.text(8, 3.5, "Final Semantic Processing Results:", 
            ha='center', fontsize=12, fontweight='bold')
    
    # P5独立块
    p5_rect = FancyBboxPatch((3, 2.5), 2.5, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor='#FFD700',
                            edgecolor='orange',
                            linewidth=2)
    ax2.add_patch(p5_rect)
    ax2.text(4.25, 2.9, "Block 5: P5\n(Low Confidence)", 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # P7+P8组合块
    p78_rect = FancyBboxPatch((10.5, 2.5), 2.5, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor='#FFD700',
                             edgecolor='orange',
                             linewidth=2)
    ax2.add_patch(p78_rect)
    ax2.text(11.75, 2.9, "Block 6: P7+P8\n(Medium Confidence)", 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 连接线显示分组过程
    # P5的处理
    ax2.annotate('', xy=(4.25, 3.3), xytext=(8, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    
    # P7+P8的处理  
    ax2.annotate('', xy=(11.75, 3.3), xytext=(12, 6.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # 关键说明
    ax2.text(8, 1.5, "Key Points:", ha='center', fontsize=11, fontweight='bold')
    ax2.text(8, 1.2, "• Only consecutive paragraphs within same window can be grouped", 
            ha='center', fontsize=10, color='blue')
    ax2.text(8, 0.9, "• P5 and P8 cannot be grouped (not in same window)", 
            ha='center', fontsize=10, color='red')
    ax2.text(8, 0.6, "• P7+P8 grouped because they are consecutive with high similarity", 
            ha='center', fontsize=10, color='green')
    ax2.text(8, 0.3, "• Window size = 5 ensures local processing efficiency", 
            ha='center', fontsize=10, color='purple')
    
    plt.tight_layout()
    return fig

def create_detailed_similarity_calculation():
    """创建详细的相似度计算演示"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.suptitle('Detailed Similarity Calculation in Sliding Windows', 
                 fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 窗口1详细分析
    ax.text(7, 11.5, "Window 1: P3, P4, P5, P6, P7 - Processing P5", 
            ha='center', fontsize=14, fontweight='bold', color='blue')
    
    # 段落内容展示
    window1_content = [
        ("P3", "Reporter: Any suggestions on management?", '#FFE6CC', True),
        ("P4", "Duan: Most important thing is culture...", '#E6F3FF', True),
        ("P5", "Background description without clear QA...", '#F0F0F0', False),
        ("P6", "Netizen A: How about Buffett's ideas?", '#FFE6CC', True),
        ("P7", "Dadao: Buffett once said about patience...", '#E6F3FF', True)
    ]
    
    for i, (para, content, color, processed) in enumerate(window1_content):
        x_pos = 1 + i * 2.4
        
        # 段落框
        rect = FancyBboxPatch((x_pos, 9.5), 2.2, 1.2,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor='black',
                             linewidth=1)
        ax.add_patch(rect)
        
        # 段落标签和内容
        ax.text(x_pos + 1.1, 10.4, para, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x_pos + 1.1, 9.8, content[:20] + "...", ha='center', va='center', fontsize=8)
        
        # 处理状态
        status = "✓ Processed" if processed else "? Process"
        ax.text(x_pos + 1.1, 9.2, status, ha='center', va='center', fontsize=8, 
               color='green' if processed else 'red')
    
    # 相似度计算
    ax.text(7, 8.5, "Similarity Calculations for P5:", ha='center', fontsize=12, fontweight='bold')
    
    # P4 ↔ P5
    ax.text(2, 7.8, "P4 ↔ P5 Similarity:", fontsize=11, fontweight='bold')
    ax.text(2, 7.5, "P4: 'Most important thing is culture and values in management'", fontsize=9)
    ax.text(2, 7.2, "P5: 'Background description without clear QA pattern here'", fontsize=9)
    ax.text(2, 6.9, "SentenceTransformer encoding → Cosine similarity = 0.42", fontsize=9, color='blue')
    ax.text(2, 6.6, "0.42 < threshold(0.55) → NOT grouped", fontsize=9, color='red')
    
    # P5 ↔ P6  
    ax.text(8, 7.8, "P5 ↔ P6 Similarity:", fontsize=11, fontweight='bold')
    ax.text(8, 7.5, "P5: 'Background description without clear QA pattern here'", fontsize=9)
    ax.text(8, 7.2, "P6: 'How about Buffett's investment ideas and philosophy'", fontsize=9)
    ax.text(8, 6.9, "SentenceTransformer encoding → Cosine similarity = 0.38", fontsize=9, color='blue')
    ax.text(8, 6.6, "0.38 < threshold(0.55) → NOT grouped", fontsize=9, color='red')
    
    # 动态阈值计算
    ax.text(7, 5.8, "Dynamic Threshold Calculation:", ha='center', fontsize=12, fontweight='bold')
    similarity_matrix = """
    Window 1 Similarities:
    P3↔P4: 0.72, P3↔P5: 0.35, P3↔P6: 0.68, P3↔P7: 0.58
    P4↔P5: 0.42, P4↔P6: 0.45, P4↔P7: 0.52
    P5↔P6: 0.38, P5↔P7: 0.41
    P6↔P7: 0.75
    
    Average similarity = 0.53
    Standard deviation = 0.14
    Dynamic threshold = 0.53 - 0.5 × 0.14 = 0.46
    
    Final threshold used = max(0.46, default_0.55) = 0.55
    """
    
    ax.text(7, 4.5, similarity_matrix, ha='center', fontsize=9, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
    
    # 窗口2分析
    ax.text(7, 2.8, "Window 2: P6, P7, P8, P9, P10 - Processing P8", 
            ha='center', fontsize=12, fontweight='bold', color='green')
    
    ax.text(7, 2.4, "P7 ↔ P8 Similarity Calculation:", ha='center', fontsize=11, fontweight='bold')
    ax.text(2, 2.0, "P7: 'Buffett once said about patience and long-term thinking in investment'", fontsize=9)
    ax.text(2, 1.7, "P8: 'Additional notes about compound interest theory and its applications'", fontsize=9)
    ax.text(2, 1.4, "SentenceTransformer encoding → Cosine similarity = 0.68", fontsize=9, color='blue')
    ax.text(2, 1.1, "0.68 > threshold(0.55) → GROUPED as P7+P8", fontsize=9, color='green')
    
    # 结果框
    result_rect = FancyBboxPatch((9, 1.5), 4, 1,
                                boxstyle="round,pad=0.1",
                                facecolor='#90EE90',
                                edgecolor='green',
                                linewidth=2)
    ax.add_patch(result_rect)
    ax.text(11, 2, "Result:\nP7+P8 Medium\nConfidence Block", 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Creating detailed remaining paragraphs grouping demonstration...")
    
    # 剩余段落分组演示
    fig1 = create_remaining_paragraphs_demo()
    fig1.savefig('/Users/skybay7/CloudStorage/BaiduYun/LLM-Prj/agent-qa-pair-generate/legend-qa-extractor/remaining_paragraphs_grouping.png',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print("✅ Remaining paragraphs grouping saved: remaining_paragraphs_grouping.png")
    
    # 详细相似度计算演示
    fig2 = create_detailed_similarity_calculation()
    fig2.savefig('/Users/skybay7/CloudStorage/BaiduYun/LLM-Prj/agent-qa-pair-generate/legend-qa-extractor/similarity_calculation_demo.png',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print("✅ Similarity calculation demo saved: similarity_calculation_demo.png")
    
    plt.show()
    print("📊 Detailed grouping demonstrations created successfully!")