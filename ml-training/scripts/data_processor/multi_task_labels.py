"""
多任务标签生成模块
为地震预测模型生成3个时间窗口×4个震级类别的标签
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

from grid_system import JapanGridSystem

class MultiTaskLabelGenerator:
    """多任务标签生成器"""

    def __init__(self, grid_system: JapanGridSystem,
                 prediction_windows: List[int] = [7, 14, 30],
                 magnitude_thresholds: List[Tuple[float, float]] = [(3.0, 4.5), (4.5, 5.5), (5.5, 6.5), (6.5, 10.0)]):
        """
        初始化标签生成器

        Args:
            grid_system: 网格系统实例
            prediction_windows: 预测时间窗口(天)
            magnitude_thresholds: 震级分组阈值 [(min, max), ...] - 修改为更均衡的分组
        """
        self.grid_system = grid_system
        self.prediction_windows = prediction_windows
        self.magnitude_thresholds = magnitude_thresholds  # 使用新的震级分组

        # 任务定义
        self.num_time_windows = len(prediction_windows)
        self.num_magnitude_classes = len(magnitude_thresholds)
        self.num_tasks = self.num_time_windows * self.num_magnitude_classes

        # 任务映射
        self.task_map = {}
        self.task_names = []

        task_idx = 0
        for time_idx, time_window in enumerate(prediction_windows):
            for mag_idx, (mag_min, mag_max) in enumerate(magnitude_thresholds):
                task_name = f"T{time_window}d_M{mag_min}-{mag_max}"
                self.task_map[task_idx] = {
                    'time_window': time_window,
                    'magnitude_range': (mag_min, mag_max),
                    'time_idx': time_idx,
                    'magnitude_idx': mag_idx,
                    'name': task_name
                }
                self.task_names.append(task_name)
                task_idx += 1

        print(f"多任务标签生成器初始化:")
        print(f"  时间窗口: {prediction_windows} 天")
        print(f"  震级分组: {magnitude_thresholds}")
        print(f"  总任务数: {self.num_tasks} ({self.num_time_windows} × {self.num_magnitude_classes})")
        print("  任务列表:")
        for i, name in enumerate(self.task_names):
            print(f"    任务{i:2d}: {name}")

    def generate_labels_for_samples(self, df: pd.DataFrame,
                                   timestamps: np.ndarray,
                                   target_region_size: Tuple[int, int] = (10, 10)) -> np.ndarray:
        """
        为所有样本生成多任务标签

        Args:
            df: 地震数据DataFrame (包含网格索引)
            timestamps: 样本时间戳数组
            target_region_size: 目标预测区域大小 (纬度网格数, 经度网格数)

        Returns:
            标签数组 [样本数, 任务数]
        """
        print(f"\n:label:  生成多任务标签...")
        print(f"  样本数: {len(timestamps)}")
        print(f"  目标区域大小: {target_region_size[0]} × {target_region_size[1]} 网格")
        print(f"  任务数: {self.num_tasks}")

        num_samples = len(timestamps)
        labels = np.zeros((num_samples, self.num_tasks), dtype=np.float32)

        # 为每个样本生成标签
        for sample_idx, timestamp in enumerate(timestamps):
            if (sample_idx + 1) % 100 == 0:
                print(f"    进度: {sample_idx+1}/{num_samples} ({(sample_idx+1)/num_samples*100:.1f}%)")

            # 生成该样本的标签
            sample_labels = self._generate_sample_labels(
                df, timestamp, target_region_size
            )
            labels[sample_idx] = sample_labels

        # 统计标签分布
        self._print_label_statistics(labels)

        print("  :white_check_mark: 多任务标签生成完成")
        return labels

    def _generate_sample_labels(self, df: pd.DataFrame,
                               timestamp: pd.Timestamp,
                               target_region_size: Tuple[int, int]) -> np.ndarray:
        """
        优化的样本标签生成 - 考虑3x3网格的特性
        """
        # 由于网格变大，目标区域也相应调整
        adjusted_region_size = (min(3, target_region_size[0]), min(3, target_region_size[1]))
        
        sample_labels = np.zeros(self.num_tasks, dtype=np.float32)
        
        for task_idx in range(self.num_tasks):
            task_info = self.task_map[task_idx]
            time_window = task_info['time_window']
            mag_min, mag_max = task_info['magnitude_range']
            
            # 定义预测时间窗口
            future_start = timestamp
            future_end = timestamp + timedelta(days=time_window)
            
            # 获取未来时间窗口内的地震
            future_earthquakes = df[
                (df['time'] >= future_start) &
                (df['time'] < future_end) &
                (df['magnitude'] >= mag_min) &
                (df['magnitude'] < mag_max)
            ]
            
            # 计算改进的概率
            prob = self._calculate_occurrence_probability(
                future_earthquakes, adjusted_region_size, time_window
            )
            
            sample_labels[task_idx] = prob
        
        return sample_labels

    def _calculate_occurrence_probability(self, earthquakes: pd.DataFrame,
                                        target_region_size: Tuple[int, int],
                                        time_window: int) -> float:
        """
        改进的地震发生概率计算 - 考虑能量释放和空间聚集性
        """
        if len(earthquakes) == 0:
            return 0.0
        
        # 1. 基于能量的概率计算（使用Gutenberg-Richter关系）
        total_energy = 0.0
        for _, eq in earthquakes.iterrows():
            # 能量计算：log10(E) = 1.5M + 4.8
            energy = 10 ** (1.5 * eq['magnitude'] + 4.8)
            total_energy += energy
        
        # 对数能量归一化
        log_energy = np.log10(total_energy + 1)
        
        # 2. 空间聚集性评分
        unique_grids = earthquakes[['lat_idx', 'lon_idx']].drop_duplicates()
        spatial_concentration = len(unique_grids) / (target_region_size[0] * target_region_size[1])
        
        # 3. 时间密度评分
        time_density = len(earthquakes) / time_window
        
        # 4. 综合概率计算（使用sigmoid函数避免极端值）
        # 为不同震级范围使用不同的权重
        magnitude_mean = earthquakes['magnitude'].mean()
        
        if magnitude_mean >= 6.5:  # 特大震
            # 特大震即使只有一个也应该有高概率
            base_score = 0.8 + 0.2 * np.tanh(log_energy / 10)
        elif magnitude_mean >= 5.5:  # 大震
            base_score = 0.4 + 0.4 * np.tanh(log_energy / 15)
        elif magnitude_mean >= 4.5:  # 中震
            base_score = 0.2 + 0.3 * np.tanh(log_energy / 20)
        else:  # 小震
            # 小震需要更多数量才有高概率
            base_score = np.tanh(time_density / 5) * 0.5
        
        # 空间调整因子
        spatial_factor = 0.5 + 0.5 * spatial_concentration
        
        # 最终概率
        final_probability = base_score * spatial_factor
        
        # 确保在[0, 1]范围内
        return np.clip(final_probability, 0.0, 1.0)

    def _print_label_statistics(self, labels: np.ndarray):
        """打印标签统计信息"""
        print(f"\n:bar_chart: 标签统计信息:")
        print(f"  标签形状: {labels.shape}")

        print("  各任务正样本比例:")
        for task_idx in range(self.num_tasks):
            task_labels = labels[:, task_idx]
            positive_ratio = np.mean(task_labels > 0) * 100
            avg_value = np.mean(task_labels)
            print(f"    {self.task_names[task_idx]:15s}: {positive_ratio:5.1f}% (平均值: {avg_value:.3f})")

        # 按时间窗口统计
        print("  按时间窗口统计:")
        for time_idx, time_window in enumerate(self.prediction_windows):
            start_idx = time_idx * self.num_magnitude_classes
            end_idx = start_idx + self.num_magnitude_classes
            window_labels = labels[:, start_idx:end_idx]
            positive_ratio = np.mean(np.any(window_labels > 0, axis=1)) * 100
            print(f"    {time_window:2d}天窗口: {positive_ratio:5.1f}% 样本有地震")

        # 按震级统计
        print("  按震级分组统计:")
        for mag_idx, (mag_min, mag_max) in enumerate(self.magnitude_thresholds):
            mag_indices = list(range(mag_idx, self.num_tasks, self.num_magnitude_classes))
            mag_labels = labels[:, mag_indices]
            positive_ratio = np.mean(np.any(mag_labels > 0, axis=1)) * 100
            print(f"    M{mag_min}-{mag_max}: {positive_ratio:5.1f}% 样本有地震")

    def create_weighted_labels(self, labels: np.ndarray,
                             earthquake_counts: np.ndarray = None) -> np.ndarray:
        """
        创建样本权重以处理类别不平衡
        
        Args:
            labels: 原始标签 [样本数, 任务数]
            earthquake_counts: 地震计数数组 [样本数]
            
        Returns:
            样本权重 [样本数, 任务数]
        """
        print("\n⚖️ 创建样本权重...")
        
        sample_weights = np.ones_like(labels, dtype=np.float32)
        
        for task_idx in range(self.num_tasks):
            task_labels = labels[:, task_idx]
            
            # 计算正负样本比例
            positive_count = np.sum(task_labels > 0)
            negative_count = np.sum(task_labels == 0)
            
            if positive_count > 0 and negative_count > 0:
                # 计算权重 (逆频率权重)
                total = len(task_labels)
                pos_weight = total / (2 * positive_count)
                neg_weight = total / (2 * negative_count)
                
                # 为每个样本分配权重
                sample_weights[:, task_idx] = np.where(
                    task_labels > 0,
                    pos_weight,
                    neg_weight
                )
                
                if task_idx < 3 or task_idx == self.num_tasks - 1:  # 只打印部分任务
                    print(f"    任务{task_idx} ({self.task_names[task_idx]}): 正样本权重={pos_weight:.2f}, 负样本权重={neg_weight:.2f}")
        
        print("  ✅ 样本权重创建完成")
        return sample_weights
    
    def get_task_info(self) -> Dict:
        """
        获取任务信息

        Returns:
            任务信息字典
        """
        return {
            'num_tasks': self.num_tasks,
            'prediction_windows': self.prediction_windows,
            'magnitude_thresholds': self.magnitude_thresholds,
            'task_map': self.task_map,
            'task_names': self.task_names
        }