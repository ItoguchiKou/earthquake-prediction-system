"""
数据聚合策略
将稀疏的网格数据聚合以提高数据密度
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，使用简单的替代
    def tqdm(iterable, desc=""):
        total = len(iterable)
        for i, item in enumerate(iterable):
            if i % max(1, total // 20) == 0:  # 每5%打印一次
                print(f"{desc}: {i}/{total} ({i/total*100:.0f}%)")
            yield item

class GridAggregationStrategy:
    """网格聚合策略"""
    
    def __init__(self, original_shape: Tuple[int, int] = (10, 8)):  # 改为10×8
        """
        初始化聚合策略
        
        Args:
            original_shape: 原始网格形状 (10, 8)
        """
        self.original_height, self.original_width = original_shape
        
        # 由于已经使用了不规则网格，聚合策略需要调整
        self.aggregation_schemes = {
            'regions': self._create_regional_aggregation(),
            'zones': self._create_seismic_zones_v2(),
            'prefectures': self._create_prefecture_based_aggregation()
        }
        
    def _create_regional_aggregation(self) -> Dict[str, List[str]]:
        """创建基于有效网格的区域聚合"""
        regions = {
            '冲绳': ['0_4'],
            '九州': ['1_2', '1_3', '2_2', '2_3', '3_2', '3_3'],
            '中国四国': ['3_4', '4_2', '4_3'],
            '近畿中部': ['3_5', '4_4', '4_5', '5_3', '5_4', '5_5'],
            '关东': ['5_6', '6_5', '6_6'],
            '东北': ['6_4', '7_4', '7_5', '7_6', '8_5', '8_6'],
            '北海道': ['9_5', '9_6', '9_7']
        }
        return regions
    def _create_prefecture_based_aggregation(self) -> Dict[str, List[str]]:
        """基于都道府县的聚合（适用于结果展示）"""
        # 这里可以根据grid_system中的prefecture信息创建映射
        pass

    def _create_seismic_zones(self) -> Dict[str, List[Tuple[int, int]]]:
        """创建地震带聚合方案 - 基于日本主要地震带"""
        zones = {
            '太平洋板块俯冲带': [],    # 东部沿海
            '菲律宾海板块带': [],      # 南部沿海
            '日本海东缘带': [],        # 西部沿海
            '内陆活动带': []           # 内陆地区
        }
        
        for i in range(self.original_height):
            for j in range(self.original_width):
                if j >= 5:  # 东部
                    zones['太平洋板块俯冲带'].append((i, j))
                elif i >= 4:  # 南部
                    zones['菲律宾海板块带'].append((i, j))
                elif j <= 2:  # 西部
                    zones['日本海东缘带'].append((i, j))
                else:  # 内陆
                    zones['内陆活动带'].append((i, j))
        
        return zones
    
    def _create_adaptive_aggregation(self) -> Dict[str, List[Tuple[int, int]]]:
        """创建自适应聚合方案 - 基于数据密度动态聚合"""
        # 这里提供一个2×3的聚合方案作为示例
        aggregated = {}
        
        # 将7×8网格聚合为2×3超级网格
        super_grid_map = {
            '超级网格_0_0': [(i, j) for i in range(0, 3) for j in range(0, 3)],
            '超级网格_0_1': [(i, j) for i in range(0, 3) for j in range(3, 5)],
            '超级网格_0_2': [(i, j) for i in range(0, 3) for j in range(5, 8)],
            '超级网格_1_0': [(i, j) for i in range(3, 7) for j in range(0, 3)],
            '超级网格_1_1': [(i, j) for i in range(3, 7) for j in range(3, 5)],
            '超级网格_1_2': [(i, j) for i in range(3, 7) for j in range(5, 8)]
        }
        
        return super_grid_map
    
    def aggregate_features(self, 
                          features: np.ndarray, 
                          scheme: str = 'regions',
                          aggregation_method: str = 'mean') -> Dict[str, np.ndarray]:
        """
        聚合特征数据
        
        Args:
            features: 原始特征 [样本数, 时间步, 高, 宽, 通道]
            scheme: 聚合方案 ('regions', 'zones', 'adaptive')
            aggregation_method: 聚合方法 ('mean', 'max', 'sum')
            
        Returns:
            聚合后的特征字典
        """
        if scheme not in self.aggregation_schemes:
            raise ValueError(f"未知的聚合方案: {scheme}")
        
        aggregation_map = self.aggregation_schemes[scheme]
        aggregated_features = {}
        
        n_samples, n_time, _, _, n_channels = features.shape
        
        for region_name, grid_indices in aggregation_map.items():
            if not grid_indices:
                continue
                
            # 提取该区域的所有网格数据
            region_data = []
            for lat_idx, lon_idx in grid_indices:
                if lat_idx < features.shape[2] and lon_idx < features.shape[3]:
                    region_data.append(features[:, :, lat_idx, lon_idx, :])
            
            if region_data:
                # 聚合数据
                region_data = np.stack(region_data, axis=2)  # [样本, 时间, 网格数, 通道]
                
                if aggregation_method == 'mean':
                    aggregated = np.mean(region_data, axis=2)
                elif aggregation_method == 'max':
                    aggregated = np.max(region_data, axis=2)
                elif aggregation_method == 'sum':
                    aggregated = np.sum(region_data, axis=2)
                else:
                    raise ValueError(f"未知的聚合方法: {aggregation_method}")
                
                aggregated_features[region_name] = aggregated
        
        return aggregated_features
    
    def aggregate_labels(self,
                        labels: np.ndarray,
                        grid_earthquake_counts: Optional[Dict[Tuple[int, int], int]] = None,
                        scheme: str = 'regions') -> Dict[str, np.ndarray]:
        """
        聚合标签数据
        
        Args:
            labels: 原始标签 [样本数, 任务数]
            grid_earthquake_counts: 每个网格的地震计数
            scheme: 聚合方案
            
        Returns:
            聚合后的标签字典
        """
        # 对于标签，我们使用OR逻辑：如果区域内任何网格有地震，则该区域有地震
        # 这是一个简化的实现，实际应用中需要根据具体情况调整
        
        aggregation_map = self.aggregation_schemes[scheme]
        
        # 由于标签已经是聚合的（整个数据集的标签），这里返回区域映射信息
        region_info = {}
        for region_name, grid_indices in aggregation_map.items():
            region_info[region_name] = {
                'grid_count': len(grid_indices),
                'grid_indices': grid_indices
            }
        
        return region_info

class FocusedPredictionStrategy:
    """聚焦预测策略 - 只预测高风险区域"""
    
    def __init__(self, historical_data: pd.DataFrame):
        """
        初始化聚焦预测策略
        
        Args:
            historical_data: 历史地震数据
        """
        self.historical_data = historical_data
        self.high_risk_zones = self._identify_high_risk_zones()
        
    def _identify_high_risk_zones(self) -> List[Dict]:
        """识别高风险区域"""
        # 基于历史数据识别高活跃度区域
        if 'lat_idx' in self.historical_data.columns and 'lon_idx' in self.historical_data.columns:
            # 统计每个网格的地震频率
            grid_counts = self.historical_data.groupby(['lat_idx', 'lon_idx']).size()
            
            # 找出地震频率最高的前20%网格
            threshold = np.percentile(grid_counts.values, 80)
            high_risk_grids = grid_counts[grid_counts >= threshold].index.tolist()
            
            # 创建高风险区域
            zones = []
            for lat_idx, lon_idx in high_risk_grids:
                zone = {
                    'center': (lat_idx, lon_idx),
                    'radius': 1,  # 包含周围网格
                    'risk_level': grid_counts[(lat_idx, lon_idx)]
                }
                zones.append(zone)
            
            return zones
        else:
            return []
    
    def create_focused_model_config(self) -> Dict:
        """创建聚焦模型配置"""
        return {
            'prediction_zones': len(self.high_risk_zones),
            'zone_details': self.high_risk_zones,
            'model_type': 'zone_specific',
            'features': {
                'use_local_features': True,
                'use_regional_features': True,
                'use_temporal_correlation': True
            }
        }

def apply_data_augmentation_strategy(features: np.ndarray, 
                                   labels: np.ndarray,
                                   augmentation_factor: int = 3,
                                   probability_threshold: float = 0.1,  # 基于概率值而非二值化
                                   max_augment_samples: int = 500,
                                   preserve_negatives: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    应用数据增强策略 - 回归版本
    
    Args:
        features: 原始特征
        labels: 原始标签（概率值）
        augmentation_factor: 增强倍数
        probability_threshold: 概率阈值（高于此值的样本才增强）
        max_augment_samples: 最大增强样本数
        preserve_negatives: 是否保留所有原始数据
        
    Returns:
        增强后的特征和标签
    """
    print(f"  开始数据增强处理...")
    print(f"  原始数据形状: {features.shape}")
    
    # 基于平均概率值找出需要增强的样本
    mean_probs = np.mean(labels, axis=1)
    augment_mask = mean_probs > probability_threshold
    augment_indices = np.where(augment_mask)[0]
    
    print(f"  找到 {len(augment_indices)} 个样本需要增强（平均概率>{probability_threshold}）")
    
    if len(augment_indices) == 0:
        print("  警告：没有找到需要增强的样本")
        return features, labels
    
    # 限制增强的样本数量
    if len(augment_indices) > max_augment_samples:
        print(f"  限制增强样本数: {len(augment_indices)} → {max_augment_samples}")
        # 优先选择高概率样本
        sorted_indices = augment_indices[np.argsort(mean_probs[augment_indices])[::-1]]
        augment_indices = sorted_indices[:max_augment_samples]
    
    # 收集增强的数据
    augmented_features_list = []
    augmented_labels_list = []
    
    print(f"  开始增强 {len(augment_indices)} 个样本，每个增强 {augmentation_factor} 倍...")
    
    for idx in tqdm(augment_indices, desc="  增强进度"):
        original_feature = features[idx]
        original_label = labels[idx]
        
        for aug_i in range(augmentation_factor):
            # 1. 时间窗口滑动（更合理的数据增强）
            if aug_i < augmentation_factor // 2:
                # 使用滑动窗口
                window_shift = (aug_i + 1) * 5  # 5, 10, 15天的滑动
                if window_shift < 30:  # 确保不超出范围
                    aug_feature = np.roll(original_feature, window_shift, axis=0)
                else:
                    aug_feature = original_feature.copy()
            else:
                aug_feature = original_feature.copy()
            
            # 2. 添加符合物理规律的噪声
            # 对不同特征通道使用不同的噪声水平
            for c in range(aug_feature.shape[-1]):
                if c in [0, 1]:  # 地震频率和震级通道
                    noise_level = 0.02
                else:  # 其他通道
                    noise_level = 0.01
                
                noise = np.random.normal(0, noise_level, aug_feature[:, :, :, c].shape)
                aug_feature[:, :, :, c] = np.clip(aug_feature[:, :, :, c] + noise, 0, 1)
            
            # 3. 标签的合理扰动（保持物理一致性）
            # 根据原始概率值决定扰动幅度
            label_noise_scale = np.where(original_label > 0.3, 0.03, 0.01)
            label_noise = np.random.normal(0, label_noise_scale, original_label.shape)
            aug_label = np.clip(original_label + label_noise, 0.0, 1.0)
            
            # 确保时间一致性
            aug_label_reshaped = aug_label.reshape(3, 4)
            for t in range(2):
                # 确保后续时间窗口概率不小于前面（软约束）
                aug_label_reshaped[t+1, :] = np.maximum(
                    aug_label_reshaped[t+1, :], 
                    aug_label_reshaped[t, :] * 0.95  # 允许5%的衰减
                )
            aug_label = aug_label_reshaped.flatten()
            
            augmented_features_list.append(aug_feature)
            augmented_labels_list.append(aug_label)
    
    # 合并原始数据和增强数据
    if augmented_features_list:
        augmented_features = np.array(augmented_features_list)
        augmented_labels = np.array(augmented_labels_list)
        
        if preserve_negatives:
            # 保留所有原始数据
            final_features = np.concatenate([features, augmented_features], axis=0)
            final_labels = np.concatenate([labels, augmented_labels], axis=0)
        else:
            # 只使用增强后的数据
            final_features = augmented_features
            final_labels = augmented_labels
        
        print(f"  ✅ 数据增强完成：")
        print(f"     原始样本: {len(features)}")
        print(f"     新增样本: {len(augmented_features)}")
        print(f"     总样本数: {len(final_features)}")
        
        # 计算概率统计
        original_mean = np.mean(labels)
        final_mean = np.mean(final_labels)
        high_prob_ratio = np.mean(final_labels > 0.3)
        
        print(f"     原始标签均值: {original_mean:.4f}")
        print(f"     增强后标签均值: {final_mean:.4f}")
        print(f"     高概率(>0.3)样本比例: {high_prob_ratio:.2%}")
        
        return final_features, final_labels
    else:
        return features, labels

# 测试代码
if __name__ == "__main__":
    # 测试网格聚合
    aggregator = GridAggregationStrategy()
    
    # 模拟特征数据
    features = np.random.randn(100, 90, 7, 8, 8)  # 100样本, 90天, 7×8网格, 8通道
    
    # 测试区域聚合
    regional_features = aggregator.aggregate_features(features, scheme='regions')
    print("区域聚合结果:")
    for region, feat in regional_features.items():
        print(f"  {region}: {feat.shape}")
    
    # 测试地震带聚合
    zone_features = aggregator.aggregate_features(features, scheme='zones')
    print("\n地震带聚合结果:")
    for zone, feat in zone_features.items():
        print(f"  {zone}: {feat.shape}")
    
    # 测试自适应聚合
    adaptive_features = aggregator.aggregate_features(features, scheme='adaptive')
    print("\n自适应聚合结果:")
    for grid, feat in adaptive_features.items():
        print(f"  {grid}: {feat.shape}")