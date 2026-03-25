import requests
import time
import subprocess
import json
import csv
import os
from datetime import datetime
import traceback
import re

class VLLMMonitor:
    def __init__(self, vllm_url="http://192.168.0.172:8000", output_file=None):
        # 修复：去除 URL 中的空格
        self.vllm_url = vllm_url.strip()
        self.metrics_url = f"{self.vllm_url}/metrics"
        
        # 设置输出文件路径
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"vllm_monitor_{timestamp}.csv"
        else:
            self.output_file = output_file
        
        self.data_buffer = []  # 内存缓冲，用于批量写入
        
        # CSV 表头
        self.csv_headers = [
            'timestamp', 'datetime',
            'vllm_running', 'vllm_waiting', 'vllm_kv_cache_usage_pct',
            'gpu_utilization_pct', 'gpu_memory_used_mb', 'gpu_memory_total_mb', 
            'gpu_memory_usage_pct', 'gpu_temperature_c', 'gpu_power_w'
        ]
        
        # 初始化 CSV 文件
        self._init_csv()
        
    def _init_csv(self):
        """初始化 CSV 文件，写入表头"""
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
            print(f"数据将保存到: {os.path.abspath(self.output_file)}")
        except Exception as e:
            print(f"Error creating CSV file: {e}")
            raise
    
    def _append_to_csv(self, row_data):
        """追加单行数据到 CSV"""
        try:
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
        except Exception as e:
            print(f"Error writing to CSV: {e}")
    
    def get_vllm_metrics(self):
        """获取 vLLM 服务指标"""
        try:
            resp = requests.get(self.metrics_url, timeout=5)
            resp.raise_for_status()
            metrics = {}
            
            for line in resp.text.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # 使用正则表达式解析 Prometheus 格式
                if 'vllm:num_requests_running{' in line:
                    match = re.search(r'} (\d+\.?\d*)', line)
                    if match:
                        metrics['running'] = float(match.group(1))
                
                elif 'vllm:num_requests_waiting{' in line:
                    match = re.search(r'} (\d+\.?\d*)', line)
                    if match:
                        metrics['waiting'] = float(match.group(1))
                
                # 注意：指标名是 gpu_cache_usage_perc
                elif 'vllm:gpu_cache_usage_perc{' in line:
                    match = re.search(r'} (\d+\.?\d*)', line)
                    if match:
                        metrics['kv_cache_usage'] = float(match.group(1)) * 100
            
            return metrics
            
        except requests.exceptions.RequestException as e:
            print(f"HTTP Error fetching vLLM metrics: {e}")
            return {}
        except Exception as e:
            print(f"Error fetching vLLM metrics: {traceback.format_exc()}")
            return {}
    
    def get_gpu_metrics(self):
        """获取 GPU 指标"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                print(f"nvidia-smi error: {result.stderr}")
                return {}
            
            stdout = result.stdout.strip()
            if not stdout:
                print("nvidia-smi returned empty output")
                return {}
            
            parts = [p.strip() for p in stdout.split(',')]
            
            if len(parts) != 5:
                print(f"Unexpected nvidia-smi output format: {stdout}")
                return {}
            
            util_str, mem_used_str, mem_total_str, temp_str, power_str = parts
            
            def safe_float(value_str, default=0.0):
                """安全地将字符串转换为 float"""
                try:
                    if re.match(r'^-?\d+\.?\d*$', value_str):
                        return float(value_str)
                    else:
                        print(f"Warning: Non-numeric GPU value '{value_str}', using default {default}")
                        return default
                except (ValueError, TypeError) as e:
                    print(f"Warning: Failed to parse GPU value '{value_str}': {e}")
                    return default
            
            util = safe_float(util_str)
            mem_used = safe_float(mem_used_str)
            mem_total = safe_float(mem_total_str)
            temp = safe_float(temp_str)
            power = safe_float(power_str)
            
            memory_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0.0
            
            return {
                'gpu_utilization': util,
                'memory_used_mb': mem_used,
                'memory_total_mb': mem_total,
                'memory_percent': memory_percent,
                'temperature': temp,
                'power_w': power
            }
            
        except subprocess.TimeoutExpired:
            print("Error: nvidia-smi command timed out")
            return {}
        except FileNotFoundError:
            print("Error: nvidia-smi not found. Is NVIDIA driver installed?")
            return {}
        except Exception as e:
            print(f"Error fetching GPU metrics: {traceback.format_exc()}")
            return {}
    
    def monitor(self, duration_seconds=300, interval=5):
        """持续监控并输出，同时保存到 CSV"""
        print(f"开始监控 {duration_seconds} 秒...")
        print(f"vLLM URL: {self.vllm_url}")
        print(f"采样间隔: {interval} 秒")
        print(f"预计采样点数: {duration_seconds // interval}")
        print("-" * 100)
        print(f"{'Time':<19} | {'Running':<7} | {'Waiting':<7} | {'KV Cache':<8} | {'GPU%':<6} | {'VRAM%':<6} | {'Temp':<5} | {'Power':<6}")
        print("-" * 100)
        
        start_time = time.time()
        sample_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                timestamp = time.time()
                datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                vllm_m = self.get_vllm_metrics()
                gpu_m = self.get_gpu_metrics()
                
                # 提取数据
                running = vllm_m.get('running', 0)
                waiting = vllm_m.get('waiting', 0)
                kv_cache = vllm_m.get('kv_cache_usage', 0)
                gpu_util = gpu_m.get('gpu_utilization', 0)
                vram_percent = gpu_m.get('memory_percent', 0)
                temp = gpu_m.get('temperature', 0)
                power = gpu_m.get('power_w', 0)
                mem_used = gpu_m.get('memory_used_mb', 0)
                mem_total = gpu_m.get('memory_total_mb', 0)
                
                # 实时显示
                print(f"{datetime_str:<19} | {running:<7.0f} | {waiting:<7.0f} | {kv_cache:<8.1f} | {gpu_util:<6.1f} | {vram_percent:<6.1f} | {temp:<5.1f} | {power:<6.1f}")
                
                # 准备 CSV 行数据
                row = [
                    timestamp,           # 时间戳（用于绘图）
                    datetime_str,        # 可读时间
                    running,
                    waiting,
                    kv_cache,
                    gpu_util,
                    mem_used,
                    mem_total,
                    vram_percent,
                    temp,
                    power
                ]
                
                # 立即写入 CSV（避免程序中断丢失数据）
                self._append_to_csv(row)
                sample_count += 1
                
                # 计算精确睡眠时间
                elapsed = time.time() - timestamp
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n\n监控被用户中断")
        finally:
            print("-" * 100)
            print(f"监控结束，共采集 {sample_count} 个样本")
            print(f"数据文件: {os.path.abspath(self.output_file)}")
            
            # 生成简单的统计信息
            self._print_statistics()
    
    def _print_statistics(self):
        """打印数据文件统计信息"""
        try:
            if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                file_size = os.path.getsize(self.output_file) / 1024  # KB
                print(f"文件大小: {file_size:.2f} KB")
                print("\n可以用以下方式绘图:")
                print(f"  1. Excel: 直接打开 {self.output_file}")
                print("  2. Python pandas:")
                print(f"     import pandas as pd")
                print(f"     df = pd.read_csv('{self.output_file}')")
                print(f"     df.plot(x='datetime', y=['gpu_utilization_pct', 'vllm_running'])")
        except Exception as e:
            pass


def quick_plot(csv_file):
    """快速绘图函数，用于监控结束后立即查看趋势"""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        df = pd.read_csv(csv_file)
        
        # 转换时间戳
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # 图1: 请求数
        ax1 = axes[0]
        ax1.plot(df['datetime'], df['vllm_running'], label='Running', color='green', linewidth=2)
        ax1.plot(df['datetime'], df['vllm_waiting'], label='Waiting', color='orange', linewidth=2)
        ax1.set_ylabel('Requests')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('vLLM Request Queue')
        
        # 图2: GPU 利用率 & KV Cache
        ax2 = axes[1]
        ax2.plot(df['datetime'], df['gpu_utilization_pct'], label='GPU Util %', color='blue', linewidth=2)
        ax2.plot(df['datetime'], df['vllm_kv_cache_usage_pct'], label='KV Cache %', color='purple', linewidth=2)
        ax2.set_ylabel('Percentage (%)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('GPU Utilization & KV Cache Usage')
        
        # 图3: 显存 & 温度
        ax3 = axes[2]
        ax3_twin = ax3.twinx()
        
        ax3.plot(df['datetime'], df['gpu_memory_usage_pct'], label='VRAM %', color='red', linewidth=2)
        ax3_twin.plot(df['datetime'], df['gpu_temperature_c'], label='Temp °C', color='brown', linewidth=2, linestyle='--')
        
        ax3.set_ylabel('VRAM Usage (%)', color='red')
        ax3_twin.set_ylabel('Temperature (°C)', color='brown')
        ax3.set_xlabel('Time')
        
        # 合并图例
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Memory Usage & Temperature')
        
        plt.tight_layout()
        
        # 保存图片
        png_file = csv_file.replace('.csv', '.png')
        plt.savefig(png_file, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存: {png_file}")
        
        plt.show()
        
    except ImportError:
        print("\n提示: 安装 pandas 和 matplotlib 可自动绘图: pip install pandas matplotlib")
    except Exception as e:
        print(f"\n绘图失败: {e}")


# 使用示例
if __name__ == "__main__":
    # 自定义输出文件名（可选）
    output = f"vllm_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    monitor = VLLMMonitor(output_file=output)
    monitor.monitor(duration_seconds=300, interval=5)  # 监控5分钟，每5秒采样
    
    # 监控结束后自动绘图（如果安装了 pandas 和 matplotlib）
    print("\n正在生成图表...")
    quick_plot(monitor.output_file)