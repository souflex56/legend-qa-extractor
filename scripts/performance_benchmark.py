#!/usr/bin/env python3
"""
性能基准测试脚本 - 对比串行vs并行处理性能
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
from dataclasses import dataclass
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config.settings import load_config
from src.processor import QAExtractionProcessor

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    mode: str
    total_time: float
    blocks_processed: int
    qa_pairs_extracted: int
    avg_time_per_block: float
    fastest_block: float
    slowest_block: float
    success_rate: float
    throughput_blocks_per_minute: float
    throughput_qa_per_minute: float

class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化基准测试器
        
        Args:
            config_path: 配置文件路径
        """
        self.base_config = load_config(config_path)
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(self, test_blocks: int = 20) -> List[BenchmarkResult]:
        """运行完整的基准测试
        
        Args:
            test_blocks: 测试的block数量
            
        Returns:
            基准测试结果列表
        """
        print("🧪 开始性能基准测试...")
        print(f"📊 测试参数：{test_blocks} blocks")
        print("="*60)
        
        # 设置测试配置
        test_config = self._create_test_config(test_blocks)
        
        # 测试1: 串行处理
        print("\n🔄 测试1: 串行处理 (batch_size=1, max_workers=1)")
        serial_result = self._run_single_test("串行处理", test_config, batch_size=1, max_workers=1)
        self.results.append(serial_result)
        
        # 测试2: 中等并行
        print("\n⚡ 测试2: 中等并行 (batch_size=4, max_workers=3)")
        parallel_result = self._run_single_test("中等并行", test_config, batch_size=4, max_workers=3)
        self.results.append(parallel_result)
        
        # 测试3: 高并行（如果用户想要）
        print("\n🚀 测试3: 高并行 (batch_size=8, max_workers=5)")
        high_parallel_result = self._run_single_test("高并行", test_config, batch_size=8, max_workers=5)
        self.results.append(high_parallel_result)
        
        # 输出对比结果
        self._print_comparison()
        
        return self.results
    
    def _create_test_config(self, test_blocks: int):
        """创建测试配置"""
        config = self.base_config
        
        # 设置为测试模式
        config.extract_ratio = min(1.0, test_blocks / 100)  # 确保至少有test_blocks个块
        config.enable_qa_filter = False  # 禁用过滤，确保数据一致性
        config.enable_llm_anchor = False  # 禁用LLM锚点生成，专注测试处理速度
        config.log_level = "WARNING"  # 减少日志输出
        
        return config
    
    def _run_single_test(self, mode: str, config, batch_size: int, max_workers: int) -> BenchmarkResult:
        """运行单个测试
        
        Args:
            mode: 测试模式名称
            config: 测试配置
            batch_size: 批处理大小
            max_workers: 最大工作线程数
            
        Returns:
            测试结果
        """
        print(f"  🔧 配置: batch_size={batch_size}, max_workers={max_workers}")
        
        # 复制配置并设置测试参数
        test_config = config
        test_config.batch_size = batch_size
        test_config.max_workers = max_workers
        test_config.output_filename = f"benchmark_{mode.replace(' ', '_')}.jsonl"
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 创建处理器并运行
            processor = QAExtractionProcessor(test_config)
            
            # 预热模型（确保公平比较）
            if not processor.llm_client.is_warmed_up():
                print("  🔥 预热模型...")
                processor.llm_client.warmup_model()
            
            # 运行处理
            print("  ⏳ 开始处理...")
            results = processor.process_pdf()
            
            # 记录结束时间
            total_time = time.time() - start_time
            
            # 获取性能统计
            perf_stats = processor.llm_client.get_performance_report()
            
            # 计算结果
            stats = results['stats']
            blocks_processed = stats['successful_blocks'] + stats['failed_blocks']
            
            result = BenchmarkResult(
                mode=mode,
                total_time=total_time,
                blocks_processed=blocks_processed,
                qa_pairs_extracted=stats['qa_pairs_extracted'],
                avg_time_per_block=total_time / blocks_processed if blocks_processed > 0 else 0,
                fastest_block=perf_stats.get('fastest_request', 0),
                slowest_block=perf_stats.get('slowest_request', 0),
                success_rate=stats['success_rate'],
                throughput_blocks_per_minute=(blocks_processed / total_time) * 60 if total_time > 0 else 0,
                throughput_qa_per_minute=(stats['qa_pairs_extracted'] / total_time) * 60 if total_time > 0 else 0
            )
            
            print(f"  ✅ 完成: {total_time:.1f}秒, {stats['qa_pairs_extracted']} Q&A pairs")
            
            return result
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            return BenchmarkResult(
                mode=mode,
                total_time=time.time() - start_time,
                blocks_processed=0,
                qa_pairs_extracted=0,
                avg_time_per_block=0,
                fastest_block=0,
                slowest_block=0,
                success_rate=0,
                throughput_blocks_per_minute=0,
                throughput_qa_per_minute=0
            )
    
    def _print_comparison(self):
        """打印性能对比结果"""
        if len(self.results) < 2:
            return
        
        print("\n" + "="*80)
        print("📊 性能基准测试结果对比")
        print("="*80)
        
        # 表头
        print(f"{'指标':<20} {'串行处理':<15} {'中等并行':<15} {'高并行':<15} {'提升幅度':<15}")
        print("-" * 80)
        
        serial = self.results[0]
        parallel = self.results[1] if len(self.results) > 1 else serial
        high_parallel = self.results[2] if len(self.results) > 2 else parallel
        
        # 计算提升幅度
        def calc_improvement(base, improved):
            if base == 0:
                return "N/A"
            improvement = (base - improved) / base * 100
            return f"{improvement:+.1f}%"
        
        def calc_throughput_improvement(base, improved):
            if base == 0:
                return "N/A"
            improvement = (improved - base) / base * 100
            return f"{improvement:+.1f}%"
        
        # 打印对比数据
        data_rows = [
            ("总处理时间(秒)", f"{serial.total_time:.1f}", f"{parallel.total_time:.1f}", 
             f"{high_parallel.total_time:.1f}", calc_improvement(serial.total_time, parallel.total_time)),
            
            ("处理块数", f"{serial.blocks_processed}", f"{parallel.blocks_processed}", 
             f"{high_parallel.blocks_processed}", "N/A"),
            
            ("Q&A对数", f"{serial.qa_pairs_extracted}", f"{parallel.qa_pairs_extracted}", 
             f"{high_parallel.qa_pairs_extracted}", "N/A"),
            
            ("平均块处理时间(秒)", f"{serial.avg_time_per_block:.2f}", f"{parallel.avg_time_per_block:.2f}", 
             f"{high_parallel.avg_time_per_block:.2f}", calc_improvement(serial.avg_time_per_block, parallel.avg_time_per_block)),
            
            ("最快请求(秒)", f"{serial.fastest_block:.2f}", f"{parallel.fastest_block:.2f}", 
             f"{high_parallel.fastest_block:.2f}", calc_improvement(serial.fastest_block, parallel.fastest_block)),
            
            ("最慢请求(秒)", f"{serial.slowest_block:.2f}", f"{parallel.slowest_block:.2f}", 
             f"{high_parallel.slowest_block:.2f}", calc_improvement(serial.slowest_block, parallel.slowest_block)),
            
            ("成功率", f"{serial.success_rate:.1%}", f"{parallel.success_rate:.1%}", 
             f"{high_parallel.success_rate:.1%}", "N/A"),
            
            ("吞吐量(块/分钟)", f"{serial.throughput_blocks_per_minute:.1f}", f"{parallel.throughput_blocks_per_minute:.1f}", 
             f"{high_parallel.throughput_blocks_per_minute:.1f}", calc_throughput_improvement(serial.throughput_blocks_per_minute, parallel.throughput_blocks_per_minute)),
            
            ("吞吐量(Q&A/分钟)", f"{serial.throughput_qa_per_minute:.1f}", f"{parallel.throughput_qa_per_minute:.1f}", 
             f"{high_parallel.throughput_qa_per_minute:.1f}", calc_throughput_improvement(serial.throughput_qa_per_minute, parallel.throughput_qa_per_minute))
        ]
        
        for row in data_rows:
            print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]:<15} {row[4]:<15}")
        
        print("="*80)
        
        # 结论分析
        self._print_analysis()
    
    def _print_analysis(self):
        """打印性能分析结论"""
        if len(self.results) < 2:
            return
        
        serial = self.results[0]
        parallel = self.results[1]
        
        print("\n🔍 性能分析:")
        
        # 总体性能提升
        time_improvement = (serial.total_time - parallel.total_time) / serial.total_time * 100
        throughput_improvement = (parallel.throughput_qa_per_minute - serial.throughput_qa_per_minute) / serial.throughput_qa_per_minute * 100
        
        if time_improvement > 10:
            print(f"✅ 并行处理显著提升性能: 总时间减少 {time_improvement:.1f}%")
        elif time_improvement > 0:
            print(f"🟡 并行处理有轻微提升: 总时间减少 {time_improvement:.1f}%")
        else:
            print(f"❌ 并行处理没有提升: 总时间增加 {abs(time_improvement):.1f}%")
        
        # 吞吐量分析
        if throughput_improvement > 20:
            print(f"🚀 吞吐量大幅提升: {throughput_improvement:.1f}%")
        elif throughput_improvement > 0:
            print(f"📈 吞吐量有所提升: {throughput_improvement:.1f}%")
        else:
            print(f"📉 吞吐量反而下降: {abs(throughput_improvement):.1f}%")
        
        # 瓶颈分析
        if parallel.slowest_block > 30:
            print("⚠️  检测到性能瓶颈: 存在超过30秒的慢请求，可能是模型冷启动问题")
        
        if parallel.fastest_block < 2 and parallel.slowest_block > 20:
            print("🔄 检测到冷热启动混合: 建议增强模型预热机制")
        
        # 推荐配置
        if time_improvement > 15:
            print("✅ 推荐使用中等并行配置 (batch_size=4, max_workers=3)")
        elif len(self.results) > 2 and self.results[2].total_time < parallel.total_time:
            print("🚀 考虑使用高并行配置以获得更好性能")
        else:
            print("🔄 建议优化其他方面（如模型预热、网络连接）")
    
    def save_results(self, output_file: str = "benchmark_results.json"):
        """保存基准测试结果到文件"""
        results_data = []
        for result in self.results:
            results_data.append({
                'mode': result.mode,
                'total_time': result.total_time,
                'blocks_processed': result.blocks_processed,
                'qa_pairs_extracted': result.qa_pairs_extracted,
                'avg_time_per_block': result.avg_time_per_block,
                'fastest_block': result.fastest_block,
                'slowest_block': result.slowest_block,
                'success_rate': result.success_rate,
                'throughput_blocks_per_minute': result.throughput_blocks_per_minute,
                'throughput_qa_per_minute': result.throughput_qa_per_minute
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 基准测试结果已保存至: {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="QA提取器性能基准测试")
    parser.add_argument("--blocks", type=int, default=20, help="测试的block数量 (默认: 20)")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--output", default="benchmark_results.json", help="结果输出文件")
    
    args = parser.parse_args()
    
    # 运行基准测试
    benchmark = PerformanceBenchmark(args.config)
    results = benchmark.run_benchmark(args.blocks)
    
    # 保存结果
    benchmark.save_results(args.output)
    
    print("\n🎉 基准测试完成!")

if __name__ == "__main__":
    main() 