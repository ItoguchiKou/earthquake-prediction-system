"""
地震预测数据处理完整流水线
整合所有数据处理组件，生成训练就绪的数据
"""

import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入数据处理模块
from grid_system import JapanGridSystem
from grid_data_processor import GridDataProcessor
from multi_channel_features import MultiChannelFeatureBuilder
from multi_task_labels import MultiTaskLabelGenerator

class EarthquakeDataPipeline:
    """地震数据处理完整流水线"""
    
    def __init__(self, 
                 raw_data_dir: str,
                 output_dir: str = "data/processed_grid",
                 train_split_date: str = "2020-01-01",
                 verbose: bool = False):  # 添加verbose参数
        """
        初始化数据处理流水线
        
        Args:
            raw_data_dir: 原始数据目录路径 (相对或绝对路径)
            output_dir: 输出目录
            train_split_date: 训练/验证分割日期
            verbose: 是否输出详细日志
        """
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.train_split_date = pd.Timestamp(train_split_date)
        self.verbose = verbose  # 初始化verbose属性
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化组件
        self.grid_system = JapanGridSystem()
        self.grid_processor = GridDataProcessor(os.path.dirname(raw_data_dir))
        self.feature_builder = MultiChannelFeatureBuilder(self.grid_system, verbose=verbose)
        self.label_generator = MultiTaskLabelGenerator(self.grid_system)
        
        print(f"🔧 数据处理流水线初始化完成:")
        print(f"  原始数据目录: {self.raw_data_dir}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  训练/验证分割: {train_split_date}")
        print(f"  网格系统: {self.grid_system.lat_grids} × {self.grid_system.lon_grids}")
    
    def run_complete_pipeline(self, 
                            min_magnitude: float = 3.0,
                            history_days: int = 90,
                            step_days: int = 7,
                            prediction_windows: List[int] = [7, 14, 30]) -> bool:
        """
        运行完整的数据处理流水线
        
        Args:
            min_magnitude: 最小震级
            history_days: 历史数据窗口
            step_days: 滑动步长
            prediction_windows: 预测时间窗口
            
        Returns:
            处理是否成功
        """
        print("🚀 开始完整数据处理流水线...")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # 阶段1: 加载和网格化原始数据
            print("\n📂 阶段1: 加载和网格化原始数据")
            print("-" * 50)
            df_grid = self._load_and_grid_data(min_magnitude)
            
            if df_grid is None or df_grid.empty:
                print("❌ 数据加载失败")
                return False
            
            # 阶段2: 创建时间序列样本
            print("\n⏰ 阶段2: 创建时间序列样本")
            print("-" * 50)
            samples_data = self._create_time_series_samples(
                df_grid, history_days, step_days, prediction_windows
            )
            
            if samples_data is None:
                print("❌ 时间序列创建失败")
                return False
            
            # 阶段3: 分割训练和验证集
            print("\n✂️ 阶段3: 分割训练和验证集")
            print("-" * 50)
            train_data, val_data = self._split_train_validation(samples_data)
            
            # 阶段4: 保存最终数据
            print("\n💾 阶段4: 保存训练数据")
            print("-" * 50)
            self._save_training_data(train_data, val_data)
            
            # 生成数据报告
            self._generate_data_report(train_data, val_data)
            
            total_time = time.time() - start_time
            print(f"\n🎉 数据处理流水线完成! 总耗时: {total_time/60:.1f}分钟")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ 数据处理流水线失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_and_grid_data(self, min_magnitude: float) -> Optional[pd.DataFrame]:
        """加载和网格化数据"""
        # 发现数据文件
        files = self.grid_processor.discover_data_files()
        if not files:
            return None
        
        # 加载原始数据
        df = self.grid_processor.load_raw_data(files, min_magnitude)
        if df.empty:
            return None
        
        # 映射到网格
        df_grid = self.grid_processor.map_to_grids(df)
        
        print(f"  ✅ 网格化完成: {len(df_grid)} 条记录")
        return df_grid
    
    def _create_time_series_samples(self, 
                                df_grid: pd.DataFrame,
                                history_days: int,
                                step_days: int,
                                prediction_windows: List[int]) -> Optional[Dict]:
        """创建时间序列样本"""
        # 计算时间范围
        start_time = df_grid['time'].min()
        end_time = df_grid['time'].max()
        
        print(f"  数据时间范围: {start_time} ~ {end_time}")
        print(f"  历史窗口: {history_days} 天")
        print(f"  预测窗口: {prediction_windows} 天")
        print(f"  滑动步长: {step_days} 天")
        
        # 生成时间节点
        time_points = []
        current_time = start_time + timedelta(days=history_days)
        max_prediction_window = max(prediction_windows)
        
        while current_time + timedelta(days=max_prediction_window) <= end_time:
            time_points.append(current_time)
            current_time += timedelta(days=step_days)
        
        print(f"  生成样本数: {len(time_points)}")
        
        if len(time_points) == 0:
            print("❌ 无法生成有效样本")
            return None
        
        # 创建样本数据
        all_features = []
        stat_features_list = [] # 新增：统计特征
        all_labels = []
        all_timestamps = []
        
        print(f"  处理样本进度:")
        
        for i, timestamp in enumerate(time_points):
            # 调整进度显示频率
            if (i + 1) % 100 == 0 or i == 0 or i == len(time_points) - 1:
                progress = (i + 1) / len(time_points) * 100
                print(f"    {i+1}/{len(time_points)} ({progress:.1f}%)")
                
                # 只在详细模式下显示更多信息
                if self.verbose and ((i + 1) % 500 == 0 or i == 0):
                    print(f"      当前处理时间点: {timestamp}")
                    history_start = timestamp - timedelta(days=history_days)
                    print(f"      历史窗口: {history_start} ~ {timestamp}")
            
            # 构建该样本的特征
            sample_features = self._build_sample_features(
                df_grid, timestamp, history_days
            )
            
            # 构建该样本的标签
            sample_labels = self._build_sample_labels(
                df_grid, timestamp, prediction_windows
            )
            
            if sample_features is not None and sample_labels is not None:
                all_features.append(sample_features)
                all_labels.append(sample_labels)
                all_timestamps.append(timestamp)
        
        if len(all_features) == 0:
            print("❌ 未生成有效样本")
            return None
        
        # 转换为numpy数组
        features_array = np.array(all_features, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.float32)
        timestamps_array = np.array(all_timestamps)
        
        # 提取统计特征
        print("  📊 提取统计特征...")
        for i in range(len(features_array)):
            stat_features = self.feature_builder.extract_statistical_features(
                features_array[i]  # 传入单个样本，形状为 [时间, 纬度, 经度, 通道]
            )
            stat_features_list.append(stat_features)
        stat_features_array = np.array(stat_features_list, dtype=np.float32)
        
        # 创建加权标签
        print("  ⚖️ 创建加权标签...")
        weighted_labels = self.label_generator.create_weighted_labels(labels_array)
        
        print(f"  ✅ 样本创建完成:")
        print(f"    时序特征形状: {features_array.shape}")
        print(f"    统计特征形状: {stat_features_array.shape}")
        print(f"    标签形状: {labels_array.shape}")
        print(f"    加权标签形状: {weighted_labels.shape}")
        print(f"    有效样本数: {len(timestamps_array)}")
        
        return {
            'features': features_array,
            'stat_features': stat_features_array,
            'labels': labels_array,
            'weighted_labels': weighted_labels,
            'timestamps': timestamps_array
        }
    
    def _build_sample_features(self, 
                            df_grid: pd.DataFrame,
                            timestamp: pd.Timestamp,
                            history_days: int) -> Optional[np.ndarray]:
        """构建单个样本的特征"""
        # 获取历史数据
        history_start = timestamp - timedelta(days=history_days)
        history_data = df_grid[
            (df_grid['time'] >= history_start) &
            (df_grid['time'] < timestamp)
        ]
        
        # 使用多通道特征构建器
        try:
            features = self.feature_builder.build_enhanced_time_series(
                history_data, 
                history_days=history_days,
                verbose=False  # 关闭详细日志
            )
            return features
        except Exception as e:
            if self.verbose:  # 现在可以正确使用verbose属性
                print(f"⚠️ 样本特征构建失败 ({timestamp}): {e}")
            return None
    
    def _build_sample_labels(self, 
                           df_grid: pd.DataFrame,
                           timestamp: pd.Timestamp,
                           prediction_windows: List[int]) -> Optional[np.ndarray]:
        """构建单个样本的标签"""
        try:
            # 使用多任务标签生成器
            sample_labels = self.label_generator._generate_sample_labels(
                df_grid, timestamp, target_region_size=(10, 10)
            )
            return sample_labels
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 样本标签构建失败 ({timestamp}): {e}")
            return None
    
    def _split_train_validation(self, samples_data: Dict) -> Tuple[Dict, Dict]:
        """按时间分割训练和验证集"""
        features = samples_data['features']
        stat_features = samples_data['stat_features']
        labels = samples_data['labels']
        weighted_labels = samples_data['weighted_labels']
        timestamps = samples_data['timestamps']
        
        # 按时间分割
        train_mask = timestamps < self.train_split_date
        val_mask = timestamps >= self.train_split_date
        
        train_data = {
            'features': features[train_mask],
            'stat_features': stat_features[train_mask],
            'labels': labels[train_mask],
            'weighted_labels': weighted_labels[train_mask],
            'timestamps': timestamps[train_mask]
        }
        
        val_data = {
            'features': features[val_mask],
            'stat_features': stat_features[val_mask],
            'labels': labels[val_mask],
            'weighted_labels': weighted_labels[val_mask],
            'timestamps': timestamps[val_mask]
        }
        
        print(f"  训练集: {len(train_data['features'])} 样本")
        print(f"    时间范围: {train_data['timestamps'].min()} ~ {train_data['timestamps'].max()}")
        print(f"  验证集: {len(val_data['features'])} 样本")
        print(f"    时间范围: {val_data['timestamps'].min()} ~ {val_data['timestamps'].max()}")
        
        # 检查分割合理性
        if len(train_data['features']) == 0:
            print("⚠️ 训练集为空，调整分割日期")
        if len(val_data['features']) == 0:
            print("⚠️ 验证集为空，调整分割日期")
        
        return train_data, val_data
    
    def _save_training_data(self, train_data: Dict, val_data: Dict):
        """保存训练数据 - 修复时间戳问题"""
        # 保存训练集
        train_features_path = os.path.join(self.output_dir, "train_features.npy")
        train_stat_features_path = os.path.join(self.output_dir, "train_stat_features.npy")
        train_labels_path = os.path.join(self.output_dir, "train_labels.npy")
        train_weighted_labels_path = os.path.join(self.output_dir, "train_weighted_labels.npy")
        train_timestamps_path = os.path.join(self.output_dir, "train_timestamps.npy")
        
        np.save(train_features_path, train_data['features'])
        np.save(train_stat_features_path, train_data['stat_features'])
        np.save(train_labels_path, train_data['labels'])
        np.save(train_weighted_labels_path, train_data['weighted_labels'])
        
        # 修复：确保时间戳以float64格式保存
        train_timestamps_unix = np.array([
            ts.timestamp() if hasattr(ts, 'timestamp') else float(ts) 
            for ts in train_data['timestamps']
        ], dtype=np.float64)
        np.save(train_timestamps_path, train_timestamps_unix)
        
        print(f"  ✅ 训练集已保存:")
        print(f"    时序特征: {train_features_path}")
        print(f"    统计特征: {train_stat_features_path}")
        print(f"    标签: {train_labels_path}")
        print(f"    加权标签: {train_weighted_labels_path}")
        print(f"    时间戳: {train_timestamps_path} (dtype: {train_timestamps_unix.dtype})")
        
        # 保存验证集
        val_features_path = os.path.join(self.output_dir, "val_features.npy")
        val_stat_features_path = os.path.join(self.output_dir, "val_stat_features.npy")
        val_labels_path = os.path.join(self.output_dir, "val_labels.npy")
        val_weighted_labels_path = os.path.join(self.output_dir, "val_weighted_labels.npy")
        val_timestamps_path = os.path.join(self.output_dir, "val_timestamps.npy")
        
        np.save(val_features_path, val_data['features'])
        np.save(val_stat_features_path, val_data['stat_features'])
        np.save(val_labels_path, val_data['labels'])
        np.save(val_weighted_labels_path, val_data['weighted_labels'])
        
        # 修复：确保时间戳以float64格式保存
        val_timestamps_unix = np.array([
            ts.timestamp() if hasattr(ts, 'timestamp') else float(ts) 
            for ts in val_data['timestamps']
        ], dtype=np.float64)
        np.save(val_timestamps_path, val_timestamps_unix)
        
        print(f"  ✅ 验证集已保存:")
        print(f"    时序特征: {val_features_path}")
        print(f"    统计特征: {val_stat_features_path}")
        print(f"    标签: {val_labels_path}")
        print(f"    加权标签: {val_weighted_labels_path}")
        print(f"    时间戳: {val_timestamps_path} (dtype: {val_timestamps_unix.dtype})")
        
        # 保存处理配置
        config = {
            'pipeline_info': {
                'created_at': datetime.now().isoformat(),
                'train_split_date': self.train_split_date.isoformat(),
                'grid_system': self.grid_system.get_grid_info(),
                'feature_info': self.feature_builder.get_feature_info(),
                'task_info': self.label_generator.get_task_info()
            },
            'data_shapes': {
                'train_features': train_data['features'].shape,
                'train_stat_features': train_data['stat_features'].shape,
                'train_labels': train_data['labels'].shape,
                'train_weighted_labels': train_data['weighted_labels'].shape,
                'val_features': val_data['features'].shape,
                'val_stat_features': val_data['stat_features'].shape,
                'val_labels': val_data['labels'].shape,
                'val_weighted_labels': val_data['weighted_labels'].shape
            },
            'file_sizes_mb': {
                'train_features': os.path.getsize(train_features_path) / (1024*1024),
                'train_stat_features': os.path.getsize(train_stat_features_path) / (1024*1024),
                'train_labels': os.path.getsize(train_labels_path) / (1024*1024),
                'train_weighted_labels': os.path.getsize(train_weighted_labels_path) / (1024*1024),
                'val_features': os.path.getsize(val_features_path) / (1024*1024),
                'val_stat_features': os.path.getsize(val_stat_features_path) / (1024*1024),
                'val_labels': os.path.getsize(val_labels_path) / (1024*1024),
                'val_weighted_labels': os.path.getsize(val_weighted_labels_path) / (1024*1024)
            }
        }
        
        config_path = os.path.join(self.output_dir, "pipeline_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"  ✅ 配置已保存: {config_path}")
    
    def _generate_data_report(self, train_data: Dict, val_data: Dict):
        """生成数据报告"""
        report_dir = os.path.join(self.output_dir, "data_report")
        os.makedirs(report_dir, exist_ok=True)
        
        # 数据统计
        train_features = train_data['features']
        train_stat_features = train_data['stat_features']
        train_labels = train_data['labels']
        train_weighted_labels = train_data['weighted_labels']
        val_features = val_data['features']
        val_stat_features = val_data['stat_features']
        val_labels = val_data['labels']
        val_weighted_labels = val_data['weighted_labels']
        
        # 基础统计
        report = {
            'dataset_summary': {
                'total_samples': len(train_features) + len(val_features),
                'train_samples': len(train_features),
                'val_samples': len(val_features),
                'train_ratio': len(train_features) / (len(train_features) + len(val_features)),
                'feature_shape': list(train_features.shape[1:]),
                'stat_feature_shape': list(train_stat_features.shape[1:]),
                'label_shape': list(train_labels.shape[1:])
            },
            'time_coverage': {
                'train_start': str(train_data['timestamps'].min()),
                'train_end': str(train_data['timestamps'].max()),
                'val_start': str(val_data['timestamps'].min()),
                'val_end': str(val_data['timestamps'].max())
            },
            'feature_statistics': {
                'train_mean': train_features.mean(axis=(0,1,2,3)).tolist(),
                'train_std': train_features.std(axis=(0,1,2,3)).tolist(),
                'val_mean': val_features.mean(axis=(0,1,2,3)).tolist(),
                'val_std': val_features.std(axis=(0,1,2,3)).tolist()
            },
            'stat_feature_statistics': {
                'train_mean': train_stat_features.mean(axis=(0,1,2)).tolist(),
                'train_std': train_stat_features.std(axis=(0,1,2)).tolist(),
                'val_mean': val_stat_features.mean(axis=(0,1,2)).tolist(),
                'val_std': val_stat_features.std(axis=(0,1,2)).tolist()
            },
            'label_statistics': {
                'train_positive_rates': train_labels.mean(axis=0).tolist(),
                'val_positive_rates': val_labels.mean(axis=0).tolist(),
                'train_weighted_mean': train_weighted_labels.mean(axis=0).tolist(),
                'val_weighted_mean': val_weighted_labels.mean(axis=0).tolist(),
                'task_names': self.label_generator.task_names
            }
        }
        
        # 保存报告
        report_path = os.path.join(report_dir, "data_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印关键统计
        print(f"\n📊 数据报告:")
        print(f"  总样本数: {report['dataset_summary']['total_samples']:,}")
        print(f"  训练集: {report['dataset_summary']['train_samples']:,} ({report['dataset_summary']['train_ratio']:.1%})")
        print(f"  验证集: {report['dataset_summary']['val_samples']:,}")
        print(f"  时序特征维度: {report['dataset_summary']['feature_shape']}")
        print(f"  统计特征维度: {report['dataset_summary']['stat_feature_shape']}")
        print(f"  标签维度: {report['dataset_summary']['label_shape']}")
        
        print(f"\n  训练集标签分布 (正样本率):")
        for i, (task_name, rate) in enumerate(zip(report['label_statistics']['task_names'], 
                                                 report['label_statistics']['train_positive_rates'])):
            print(f"    {task_name:15s}: {rate:.3f}")
        
        print(f"\n  数据报告已保存: {report_path}")
    
    def verify_output_files(self) -> bool:
        """验证输出文件的完整性"""
        print("\n🔍 验证输出文件...")
        
        required_files = [
            "train_features.npy",
            "train_stat_features.npy",
            "train_labels.npy",
            "train_weighted_labels.npy", 
            "train_timestamps.npy",
            "val_features.npy",
            "val_stat_features.npy",
            "val_labels.npy",
            "val_weighted_labels.npy",
            "val_timestamps.npy"
        ]
        
        all_exists = True
        for filename in required_files:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / (1024*1024)
                print(f"  ✅ {filename}: {file_size:.1f} MB")
            else:
                print(f"  ❌ {filename}: 文件不存在")
                all_exists = False
        
        if all_exists:
            print("  ✅ 所有文件验证通过")
            
            # 验证数据形状一致性
            try:
                train_features = np.load(os.path.join(self.output_dir, "train_features.npy"))
                train_stat_features = np.load(os.path.join(self.output_dir, "train_stat_features.npy"))
                train_labels = np.load(os.path.join(self.output_dir, "train_labels.npy"))
                train_weighted_labels = np.load(os.path.join(self.output_dir, "train_weighted_labels.npy"))
                val_features = np.load(os.path.join(self.output_dir, "val_features.npy"))
                val_stat_features = np.load(os.path.join(self.output_dir, "val_stat_features.npy"))
                val_labels = np.load(os.path.join(self.output_dir, "val_labels.npy"))
                val_weighted_labels = np.load(os.path.join(self.output_dir, "val_weighted_labels.npy"))
                
                print(f"  📊 数据形状验证:")
                print(f"    训练时序特征: {train_features.shape}")
                print(f"    训练统计特征: {train_stat_features.shape}")
                print(f"    训练标签: {train_labels.shape}")
                print(f"    训练加权标签: {train_weighted_labels.shape}")
                print(f"    验证时序特征: {val_features.shape}")
                print(f"    验证统计特征: {val_stat_features.shape}")
                print(f"    验证标签: {val_labels.shape}")
                print(f"    验证加权标签: {val_weighted_labels.shape}")
                
                # 检查形状一致性
                assert train_features.shape[1:] == val_features.shape[1:], "时序特征形状不一致"
                assert train_stat_features.shape[1:] == val_stat_features.shape[1:], "统计特征形状不一致"
                assert train_labels.shape[1:] == val_labels.shape[1:], "标签形状不一致"
                assert train_weighted_labels.shape[1:] == val_weighted_labels.shape[1:], "加权标签形状不一致"
                assert train_features.shape[0] == train_labels.shape[0], "训练集特征标签数量不匹配"
                assert val_features.shape[0] == val_labels.shape[0], "验证集特征标签数量不匹配"
                
                print("  ✅ 数据形状验证通过")
                
            except Exception as e:
                print(f"  ❌ 数据验证失败: {e}")
                all_exists = False
        
        return all_exists


def main():
    """主函数"""
    print("🌍 地震预测数据处理完整流水线")
    print("=" * 80)
    
    # 配置参数 - 使用相对路径
    config = {
        'raw_data_dir': "../../data/raw",          # 相对路径：当前目录下的data/raw
        'output_dir': "../../data/processed_grid",       # 相对路径：当前目录下的data/processed_grid
        'train_split_date': "2020-01-01",          # 2020年前作为训练集
        'min_magnitude': 3.0,                      # 最小震级
        'history_days': 90,                        # 90天历史窗口
        'step_days': 7,                            # 7天步长
        'prediction_windows': [7, 14, 30]          # 预测时间窗口
    }
    
    print("🔧 流水线配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 检查原始数据目录
    if not os.path.exists(config['raw_data_dir']):
        print(f"❌ 原始数据目录不存在: {config['raw_data_dir']}")
        return False
    
    # 创建流水线
    pipeline = EarthquakeDataPipeline(
        raw_data_dir=config['raw_data_dir'],
        output_dir=config['output_dir'],
        train_split_date=config['train_split_date'],
        verbose=True  # 设置详细日志级别
    )
    
    # 运行完整流水线
    success = pipeline.run_complete_pipeline(
        min_magnitude=config['min_magnitude'],
        history_days=config['history_days'],
        step_days=config['step_days'],
        prediction_windows=config['prediction_windows']
    )
    
    if success:
        # 验证输出文件
        verification_passed = pipeline.verify_output_files()
        
        if verification_passed:
            print("\n🎉 数据处理流水线成功完成!")
            print("✅ 所有文件已生成并验证通过")
            print(f"📁 输出目录: {pipeline.output_dir}")
            return True
        else:
            print("\n❌ 文件验证失败")
            return False
    else:
        print("\n❌ 数据处理流水线失败")
        return False


def create_quick_test_pipeline(test_data_dir: str = "test_data"):
    """创建快速测试流水线 (用于验证代码)"""
    print("🧪 创建快速测试流水线...")
    
    # 创建测试数据目录
    os.makedirs(f"{test_data_dir}/raw", exist_ok=True)
    
    # 生成模拟数据
    print("📝 生成模拟测试数据...")
    
    # 模拟CSV数据
    dates = pd.date_range('2019-01-01', '2021-12-31', freq='D')
    mock_earthquakes = []
    
    np.random.seed(42)
    for date in dates[::7]:  # 每周一些地震
        n_quakes = np.random.poisson(2)  # 平均每周2个地震
        
        for _ in range(n_quakes):
            # 日本范围内的随机位置
            lat = np.random.uniform(30.0, 40.0)
            lon = np.random.uniform(135.0, 145.0)
            mag = np.random.exponential(1.5) + 3.0  # 指数分布 + 最小震级
            depth = np.random.uniform(5, 100)
            
            earthquake_time = date + timedelta(
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            mock_earthquakes.append({
                'magnitude': min(mag, 8.0),  # 限制最大震级
                'longitude': lon,
                'latitude': lat,
                'depth': depth,
                'time': earthquake_time.isoformat(),
                'place': f"Test location {lat:.1f}, {lon:.1f}",
                'mag_type': 'test',
                'id': f'test_{len(mock_earthquakes)}',
                'url': ''
            })
    
    # 保存测试数据
    df_test = pd.DataFrame(mock_earthquakes)
    test_file = f"{test_data_dir}/raw/rawData_2019-2021.csv"
    df_test.to_csv(test_file, index=False)
    
    print(f"  ✅ 测试数据已生成: {test_file}")
    print(f"     记录数: {len(df_test)}")
    print(f"     时间范围: {df_test['time'].min()} ~ {df_test['time'].max()}")
    
    # 运行测试流水线
    pipeline = EarthquakeDataPipeline(
        raw_data_dir=f"{test_data_dir}/raw",
        output_dir=f"{test_data_dir}/processed_grid",
        train_split_date="2020-07-01",
        verbose=False
    )
    
    success = pipeline.run_complete_pipeline(
        min_magnitude=3.0,
        history_days=30,    # 减少历史天数用于测试
        step_days=14,       # 增大步长减少样本数
        prediction_windows=[7, 14]  # 减少预测窗口
    )
    
    if success:
        pipeline.verify_output_files()
    
    return success


if __name__ == "__main__":
    # 选择运行模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 测试模式
        print("🧪 运行测试模式...")
        create_quick_test_pipeline()
    else:
        # 正常模式
        main()