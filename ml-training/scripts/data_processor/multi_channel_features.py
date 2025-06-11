"""
多通道特征构建模块
为CNN模型构建多维度地震特征通道 - 适配不规则网格
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, Optional
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

from grid_system import JapanGridSystem

class MultiChannelFeatureBuilder:
    """多通道特征构建器 - 不规则网格版本"""

    def __init__(self, grid_system: JapanGridSystem, verbose: bool = True):
        """
        初始化特征构建器
        
        Args:
            grid_system: 网格系统实例
            verbose: 是否输出详细日志
        """
        self.grid_system = grid_system
        self.verbose = verbose

        # 特征通道定义
        self.channels = {
            0: "earthquake_frequency",     # 地震频率密度
            1: "magnitude_mean",          # 平均震级
            2: "magnitude_max",           # 最大震级
            3: "energy_release",          # 能量释放
            4: "magnitude_std",           # 震级标准差
            5: "depth_mean",              # 平均深度
            6: "temporal_density",        # 时间密度变化
            7: "spatial_correlation"      # 空间相关性
        }

        self.num_channels = len(self.channels)
        
        # 获取有效网格列表
        self.valid_grids = grid_system.get_all_valid_grids()
        
        if self.verbose:
            print(f"多通道特征构建器初始化:")
            print(f"  有效网格数: {len(self.valid_grids)}")
            print(f"  通道数: {self.num_channels}")
            for idx, name in self.channels.items():
                print(f"    通道{idx}: {name}")

    def build_enhanced_time_series(self, df: pd.DataFrame,
                                history_days: int = 90,
                                time_step_hours: int = 24,
                                verbose: Optional[bool] = None) -> np.ndarray:
        """
        构建增强的多通道时间序列特征 - 使用密集矩阵表示
        
        Args:
            df: 网格化地震数据DataFrame
            history_days: 历史数据天数
            time_step_hours: 时间步长(小时)
            verbose: 是否输出详细日志（None时使用实例默认值）
        
        Returns:
            多通道时间序列 [时间步长, 10, 8, 通道数] - 保持兼容性
        """
        show_log = self.verbose if verbose is None else verbose
        
        if show_log and not hasattr(self, '_first_run_done'):
            print(f"\n⚙️ 构建增强多通道特征...")
            print(f"  历史天数: {history_days} 天")
            print(f"  时间步长: {time_step_hours} 小时")
            print(f"  输出维度: [{history_days}, 10, 8, {self.num_channels}]")
            self._first_run_done = True
            
        # 初始化特征数组 - 使用10×8的密集表示
        time_steps = history_days
        features = np.zeros((time_steps, 10, 8, self.num_channels))

        if len(df) == 0:
            return features

        # 按日期分组处理
        df_copy = df.copy()
        df_copy['date'] = df_copy['time'].dt.date
        start_date = df_copy['time'].min().date()

        # 为每一天构建特征
        for day_offset in range(time_steps):
            current_date = start_date + timedelta(days=day_offset)
            day_data = df_copy[df_copy['date'] == current_date]

            if len(day_data) > 0:
                # 构建当天的多通道特征
                daily_features = self._build_daily_features(day_data, df_copy, current_date)
                features[day_offset] = daily_features

        # 后处理：平滑和归一化
        features = self._post_process_features(features)

        return features

    def _build_daily_features(self, day_data: pd.DataFrame,
                            all_data: pd.DataFrame,
                            current_date) -> np.ndarray:
        """
        构建单日多通道特征 - 只处理有效网格

        Args:
            day_data: 当日地震数据
            all_data: 所有历史数据
            current_date: 当前日期

        Returns:
            单日特征 [10, 8, 通道数] - 无效网格填充0
        """
        daily_features = np.zeros((10, 8, self.num_channels))

        # 按网格分组
        grid_groups = day_data.groupby(['lat_idx', 'lon_idx'])

        for (lat_idx, lon_idx), grid_data in grid_groups:
            lat_idx, lon_idx = int(lat_idx), int(lon_idx)
            
            # 检查是否是有效网格
            if (lat_idx, lon_idx) in self.valid_grids:
                # 构建该网格的特征
                grid_features = self._build_grid_features(
                    grid_data, all_data, current_date, lat_idx, lon_idx
                )
                daily_features[lat_idx, lon_idx] = grid_features

        return daily_features

    def _build_grid_features(self, grid_data: pd.DataFrame,
                           all_data: pd.DataFrame,
                           current_date,
                           lat_idx: int,
                           lon_idx: int) -> np.ndarray:
        """
        构建单个网格的特征向量 - 增强版本
        """
        features = np.zeros(self.num_channels)
        
        # 基础统计特征
        magnitudes = grid_data['magnitude'].values
        depths = grid_data['depth'].values if 'depth' in grid_data.columns else np.array([0])
        
        if len(magnitudes) > 0:
            # 通道0: 地震频率密度（对数变换减少稀疏性）
            features[0] = np.log1p(len(grid_data))
            
            # 通道1: 平均震级（加权平均，大震权重更高）
            weights = np.exp(magnitudes - magnitudes.min())
            features[1] = np.average(magnitudes, weights=weights)
            
            # 通道2: 最大震级
            features[2] = np.max(magnitudes)
            
            # 通道3: 能量释放（使用更准确的能量公式）
            energies = 10 ** (1.5 * magnitudes + 4.8)
            features[3] = np.log10(np.sum(energies) + 1)
            
            # 通道4: 震级标准差（衡量地震活动的多样性）
            features[4] = np.std(magnitudes) if len(magnitudes) > 1 else 0
            
            # 通道5: 平均深度（归一化到0-1）
            features[5] = np.mean(depths) / 100.0 if len(depths) > 0 else 0
            
            # 通道6: 时间密度变化（改进的计算方法）
            features[6] = self._calculate_improved_temporal_density(
                all_data, current_date, lat_idx, lon_idx
            )
            
            # 通道7: 空间相关性（只考虑有效邻域）
            features[7] = self._calculate_improved_spatial_correlation(
                all_data, current_date, lat_idx, lon_idx
            )
        
        return features
    
    def _calculate_improved_temporal_density(self, all_data: pd.DataFrame,
                                          current_date,
                                          lat_idx: int,
                                          lon_idx: int) -> float:
        """
        改进的时间密度计算 - 使用指数加权移动平均
        """
        try:
            # 获取过去30天的数据
            time_windows = [7, 14, 30]  # 多个时间窗口
            density_scores = []
            
            for window in time_windows:
                window_start = current_date - timedelta(days=window)
                window_data = all_data[
                    (all_data['lat_idx'] == lat_idx) &
                    (all_data['lon_idx'] == lon_idx) &
                    (all_data['date'] >= window_start) &
                    (all_data['date'] < current_date)
                ]
                
                # 使用指数加权（最近的事件权重更高）
                if len(window_data) > 0:
                    days_ago = (current_date - window_data['date']).dt.days
                    weights = np.exp(-days_ago / window)
                    weighted_count = np.sum(weights)
                    density_scores.append(weighted_count / window)
                else:
                    density_scores.append(0.0)
            
            # 综合多个时间窗口的密度
            return np.mean(density_scores)
            
        except:
            return 0.0
        
    def _calculate_improved_spatial_correlation(self, all_data: pd.DataFrame,
                                             current_date,
                                             lat_idx: int,
                                             lon_idx: int) -> float:
        """
        改进的空间相关性计算 - 只考虑有效邻域网格
        """
        try:
            # 获取有效邻域网格
            neighbors = self.grid_system.get_neighbor_grids(lat_idx, lon_idx, radius=1)
            
            # 过去30天的数据
            recent_data = all_data[
                (all_data['date'] >= current_date - timedelta(days=30)) &
                (all_data['date'] <= current_date)
            ]
            
            if len(recent_data) == 0:
                return 0.0
            
            # 计算中心网格的活动强度
            center_activity = len(recent_data[
                (recent_data['lat_idx'] == lat_idx) &
                (recent_data['lon_idx'] == lon_idx)
            ])
            
            # 计算邻域活动强度（距离加权）
            neighbor_activities = []
            for nlat, nlon in neighbors:
                if nlat == lat_idx and nlon == lon_idx:
                    continue
                
                # 计算距离
                distance = np.sqrt((nlat - lat_idx)**2 + (nlon - lon_idx)**2)
                weight = 1.0 / (1.0 + distance)
                
                neighbor_count = len(recent_data[
                    (recent_data['lat_idx'] == nlat) &
                    (recent_data['lon_idx'] == nlon)
                ])
                
                neighbor_activities.append(neighbor_count * weight)
            
            # 空间相关性分数
            if neighbor_activities:
                avg_neighbor_activity = np.mean(neighbor_activities)
                correlation = center_activity / (avg_neighbor_activity + 1.0)
                return np.tanh(correlation)  # 归一化到[-1, 1]
            else:
                return 0.0
                
        except:
            return 0.0

    def _post_process_features(self, features: np.ndarray) -> np.ndarray:
        """
        特征后处理：平滑和归一化 - 只处理有效网格

        Args:
            features: 原始特征 [时间步长, 10, 8, 通道数]

        Returns:
            处理后特征
        """
        processed_features = features.copy()
        
        # 创建有效网格掩码
        valid_mask = np.zeros((10, 8), dtype=bool)
        for lat_idx, lon_idx in self.valid_grids:
            valid_mask[lat_idx, lon_idx] = True
        
        # 对每个通道进行处理
        for channel in range(self.num_channels):
            channel_data = processed_features[:, :, :, channel]
            
            # 1. 对稀疏通道进行空间平滑（只在有效网格内）
            if self.channels[channel] in ["earthquake_frequency", "energy_release"]:
                for t in range(len(channel_data)):
                    frame = channel_data[t]
                    # 只考虑有效网格的非零值
                    valid_nonzero = np.sum((frame > 0) & valid_mask)
                    valid_total = np.sum(valid_mask)
                    
                    if valid_nonzero < 0.1 * valid_total:  # 如果有效网格中非零值少于10%
                        # 使用高斯平滑，但保持无效网格为0
                        smoothed = ndimage.gaussian_filter(frame, sigma=1.0)
                        smoothed[~valid_mask] = 0  # 无效网格置0
                        processed_features[t, :, :, channel] = smoothed
            
            # 2. 自适应归一化（只基于有效网格的值）
            channel_data = processed_features[:, :, :, channel]
            valid_data = channel_data[valid_mask.reshape(1, 10, 8).repeat(len(channel_data), axis=0)]
            
            if len(valid_data) > 0 and np.max(valid_data) > 0:
                # 使用有效数据的百分位数
                p95 = np.percentile(valid_data[valid_data > 0], 95)
                p5 = np.percentile(valid_data[valid_data > 0], 5)
                
                if p95 > p5:
                    # 裁剪极端值
                    channel_data = np.clip(channel_data, 0, p95)  # 保持下限为0
                    # 归一化到[0, 1]，无效网格保持为0
                    normalized = (channel_data - p5) / (p95 - p5)
                    normalized[~valid_mask] = 0
                    processed_features[:, :, :, channel] = normalized
                else:
                    # 如果数据范围太小，使用标准归一化
                    if np.max(valid_data) > 0:
                        normalized = channel_data / np.max(valid_data)
                        normalized[~valid_mask] = 0
                        processed_features[:, :, :, channel] = normalized
        
        return processed_features

    def extract_statistical_features(self, time_series: np.ndarray) -> np.ndarray:
        """
        从时间序列中提取统计特征 - 只处理有效网格

        Args:
            time_series: 输入时间序列 [时间步长, 10, 8, 通道数]

        Returns:
            统计特征 [10, 8, 统计特征数] - 无效网格填充0
        """
        if self.verbose and not hasattr(self, '_stat_features_logged'):
            print("\n📊 提取统计特征...")
            self._stat_features_logged = True

        time_steps, _, _, channels = time_series.shape

        # 统计特征类型
        stat_features = {
            'mean': 0,
            'std': 1,
            'max': 2,
            'min': 3,
            'trend': 4,     # 线性趋势
            'volatility': 5  # 波动性
        }

        num_stat_features = len(stat_features)
        total_features = channels * num_stat_features

        statistical_features = np.zeros((10, 8, total_features))

        # 只处理有效网格
        for lat_idx, lon_idx in self.valid_grids:
            for ch in range(channels):
                # 提取该网格该通道的时间序列
                ts = time_series[:, lat_idx, lon_idx, ch]

                base_idx = ch * num_stat_features

                # 基础统计特征
                statistical_features[lat_idx, lon_idx, base_idx + 0] = np.mean(ts)
                statistical_features[lat_idx, lon_idx, base_idx + 1] = np.std(ts)
                statistical_features[lat_idx, lon_idx, base_idx + 2] = np.max(ts)
                statistical_features[lat_idx, lon_idx, base_idx + 3] = np.min(ts)

                # 趋势特征 (线性回归斜率)
                if np.std(ts) > 1e-6:
                    x = np.arange(len(ts))
                    trend = np.polyfit(x, ts, 1)[0]
                    statistical_features[lat_idx, lon_idx, base_idx + 4] = trend

                # 波动性特征 (变异系数)
                mean_val = np.mean(ts)
                if mean_val > 1e-6:
                    volatility = np.std(ts) / mean_val
                    statistical_features[lat_idx, lon_idx, base_idx + 5] = volatility

        if self.verbose and not hasattr(self, '_stat_features_shape_logged'):
            print(f"  ✅ 统计特征提取完成: {statistical_features.shape}")
            self._stat_features_shape_logged = True
            
        return statistical_features

    def get_feature_info(self) -> Dict:
        """
        获取特征信息

        Returns:
            特征信息字典
        """
        return {
            'num_channels': self.num_channels,
            'channel_names': self.channels,
            'grid_shape': (10, 8),  # 保持兼容性
            'num_valid_grids': len(self.valid_grids),
            'valid_grids': self.valid_grids,
            'feature_description': {
                0: "地震频率密度 - 反映地震活跃程度",
                1: "平均震级 - 反映地震强度水平",
                2: "最大震级 - 反映最强地震强度",
                3: "能量释放 - 基于Gutenberg-Richter关系",
                4: "震级标准差 - 反映震级分布的离散程度",
                5: "平均深度 - 反映震源深度特征",
                6: "时间密度变化 - 反映活跃度变化趋势",
                7: "空间相关性 - 反映与邻域的关联性"
            }
        }