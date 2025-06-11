"""
数据验证脚本
检查原始数据和增强后数据的完整性和正确性
"""

import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    """数据验证器"""
    
    def __init__(self, data_dir: str, is_augmented: bool = False):
        """
        初始化验证器
        
        Args:
            data_dir: 数据目录
            is_augmented: 是否是增强后的数据
        """
        self.data_dir = data_dir
        self.is_augmented = is_augmented
        self.suffix = '_aug' if is_augmented else ''
        self.report = {}
        
    def validate_all(self, save_report: bool = True) -> Dict:
        """执行所有验证"""
        print(f"🔍 开始验证数据: {self.data_dir}")
        print("="*60)
        
        # 1. 检查文件存在性
        print("\n1. 检查文件存在性...")
        self.check_file_existence()
        
        # 2. 检查数据形状
        print("\n2. 检查数据形状...")
        self.check_data_shapes()
        
        # 3. 检查数据值范围
        print("\n3. 检查数据值范围...")
        self.check_value_ranges()
        
        # 4. 检查标签分布
        print("\n4. 检查标签分布...")
        self.check_label_distribution()
        
        # 5. 检查时间戳
        print("\n5. 检查时间戳...")
        self.check_timestamps()
        
        # 6. 检查数据一致性
        print("\n6. 检查数据一致性...")
        self.check_data_consistency()
        
        # 7. 检查内存使用
        print("\n7. 检查内存使用...")
        self.check_memory_usage()
        
        # 8. 生成可视化
        print("\n8. 生成可视化...")
        self.generate_visualizations()
        
        # 保存报告
        if save_report:
            report_path = os.path.join(self.data_dir, f'validation_report{self.suffix}.json')
            with open(report_path, 'w') as f:
                json.dump(self.report, f, indent=2, default=str)
            print(f"\n📄 验证报告已保存: {report_path}")
        
        return self.report
    
    def check_file_existence(self):
        """检查文件是否存在"""
        splits = ['train', 'val']
        file_types = ['features', 'stat_features', 'labels', 'timestamps']
        
        self.report['file_existence'] = {}
        all_exist = True
        
        for split in splits:
            self.report['file_existence'][split] = {}
            for file_type in file_types:
                filename = f"{split}_{file_type}{self.suffix}.npy"
                filepath = os.path.join(self.data_dir, filename)
                exists = os.path.exists(filepath)
                self.report['file_existence'][split][file_type] = exists
                
                if exists:
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"  ✓ {filename}: {size_mb:.1f} MB")
                else:
                    print(f"  ✗ {filename}: 不存在")
                    all_exist = False
        
        self.report['all_files_exist'] = all_exist
        
    def check_data_shapes(self):
        """检查数据形状"""
        self.report['data_shapes'] = {}
        
        for split in ['train', 'val']:
            print(f"\n  {split}集:")
            self.report['data_shapes'][split] = {}
            
            try:
                # 加载数据
                features = np.load(os.path.join(self.data_dir, f"{split}_features{self.suffix}.npy"), mmap_mode='r')
                stat_features = np.load(os.path.join(self.data_dir, f"{split}_stat_features{self.suffix}.npy"), mmap_mode='r')
                labels = np.load(os.path.join(self.data_dir, f"{split}_labels{self.suffix}.npy"), mmap_mode='r')
                timestamps = np.load(os.path.join(self.data_dir, f"{split}_timestamps{self.suffix}.npy"), mmap_mode='r')
                
                # 记录形状
                shapes = {
                    'features': features.shape,
                    'stat_features': stat_features.shape,
                    'labels': labels.shape,
                    'timestamps': timestamps.shape
                }
                
                self.report['data_shapes'][split] = {k: list(v) for k, v in shapes.items()}
                
                # 打印形状
                print(f"    features: {features.shape}")
                print(f"    stat_features: {stat_features.shape}")
                print(f"    labels: {labels.shape}")
                print(f"    timestamps: {timestamps.shape}")
                
                # 检查一致性
                n_samples = features.shape[0]
                consistent = all([
                    stat_features.shape[0] == n_samples,
                    labels.shape[0] == n_samples,
                    timestamps.shape[0] == n_samples
                ])
                
                if consistent:
                    print(f"    ✓ 样本数一致: {n_samples}")
                else:
                    print(f"    ✗ 样本数不一致!")
                
                self.report['data_shapes'][split]['consistent'] = consistent
                self.report['data_shapes'][split]['n_samples'] = int(n_samples)
                
            except Exception as e:
                print(f"    ❌ 加载失败: {e}")
                self.report['data_shapes'][split]['error'] = str(e)
    
    def check_value_ranges(self):
        """检查数据值范围"""
        self.report['value_ranges'] = {}
        
        for split in ['train', 'val']:
            print(f"\n  {split}集:")
            self.report['value_ranges'][split] = {}
            
            try:
                # 加载小批量数据进行检查
                features = np.load(os.path.join(self.data_dir, f"{split}_features{self.suffix}.npy"), mmap_mode='r')
                stat_features = np.load(os.path.join(self.data_dir, f"{split}_stat_features{self.suffix}.npy"), mmap_mode='r')
                labels = np.load(os.path.join(self.data_dir, f"{split}_labels{self.suffix}.npy"), mmap_mode='r')
                
                # 采样检查（最多1000个样本）
                n_check = min(1000, len(features))
                indices = np.random.choice(len(features), n_check, replace=False)
                
                # 特征值范围
                feat_sample = features[indices]
                feat_stats = {
                    'min': float(np.min(feat_sample)),
                    'max': float(np.max(feat_sample)),
                    'mean': float(np.mean(feat_sample)),
                    'std': float(np.std(feat_sample))
                }
                print(f"    特征值: [{feat_stats['min']:.3f}, {feat_stats['max']:.3f}], "
                      f"均值={feat_stats['mean']:.3f}, 标准差={feat_stats['std']:.3f}")
                
                # 统计特征范围
                stat_sample = stat_features[indices]
                stat_stats = {
                    'min': float(np.min(stat_sample)),
                    'max': float(np.max(stat_sample)),
                    'mean': float(np.mean(stat_sample)),
                    'std': float(np.std(stat_sample))
                }
                print(f"    统计特征: [{stat_stats['min']:.3f}, {stat_stats['max']:.3f}], "
                      f"均值={stat_stats['mean']:.3f}, 标准差={stat_stats['std']:.3f}")
                
                # 标签范围
                label_sample = labels[indices]
                label_stats = {
                    'min': float(np.min(label_sample)),
                    'max': float(np.max(label_sample)),
                    'unique_values': sorted(list(np.unique(label_sample)))[:10]  # 最多显示10个
                }
                print(f"    标签值: [{label_stats['min']:.3f}, {label_stats['max']:.3f}]")
                
                # 检查异常值
                has_nan = bool(np.any(np.isnan(feat_sample)) or np.any(np.isnan(stat_sample)))
                has_inf = bool(np.any(np.isinf(feat_sample)) or np.any(np.isinf(stat_sample)))
                
                if has_nan:
                    print(f"    ⚠️  包含NaN值!")
                if has_inf:
                    print(f"    ⚠️  包含Inf值!")
                
                self.report['value_ranges'][split] = {
                    'features': feat_stats,
                    'stat_features': stat_stats,
                    'labels': label_stats,
                    'has_nan': has_nan,
                    'has_inf': has_inf
                }
                
            except Exception as e:
                print(f"    ❌ 检查失败: {e}")
                self.report['value_ranges'][split]['error'] = str(e)
    
    def check_label_distribution(self):
        """检查标签分布"""
        self.report['label_distribution'] = {}
        
        task_names = [
            "7天_M3.0-4.5", "7天_M4.5-5.5", "7天_M5.5-6.5", "7天_M6.5+",
            "14天_M3.0-4.5", "14天_M4.5-5.5", "14天_M5.5-6.5", "14天_M6.5+",
            "30天_M3.0-4.5", "30天_M4.5-5.5", "30天_M5.5-6.5", "30天_M6.5+"
        ]
        
        for split in ['train', 'val']:
            print(f"\n  {split}集标签分布:")
            self.report['label_distribution'][split] = {}
            
            try:
                labels = np.load(os.path.join(self.data_dir, f"{split}_labels{self.suffix}.npy"))
                
                # 整体正样本率
                positive_mask = np.any(labels > 0.5, axis=1)
                overall_positive_ratio = np.mean(positive_mask)
                print(f"    整体正样本率: {overall_positive_ratio:.2%} ({np.sum(positive_mask)}/{len(labels)})")
                
                # 每个任务的正样本率
                task_stats = []
                print(f"    任务级别正样本率:")
                for i in range(labels.shape[1]):
                    positive_count = np.sum(labels[:, i] > 0.5)
                    positive_ratio = np.mean(labels[:, i] > 0.5)
                    
                    task_stat = {
                        'task_id': i,
                        'task_name': task_names[i] if i < len(task_names) else f"任务{i}",
                        'positive_count': int(positive_count),
                        'positive_ratio': float(positive_ratio)
                    }
                    task_stats.append(task_stat)
                    
                    if positive_ratio > 0:
                        print(f"      任务{i:2d} ({task_names[i]:12s}): {positive_ratio:6.2%} ({positive_count:5d}个)")
                    else:
                        print(f"      任务{i:2d} ({task_names[i]:12s}): {positive_ratio:6.2%} (无正样本)")
                
                self.report['label_distribution'][split] = {
                    'overall_positive_ratio': float(overall_positive_ratio),
                    'overall_positive_count': int(np.sum(positive_mask)),
                    'task_stats': task_stats
                }
                
            except Exception as e:
                print(f"    ❌ 检查失败: {e}")
                self.report['label_distribution'][split]['error'] = str(e)
    
    def check_timestamps(self):
        """检查时间戳"""
        self.report['timestamps'] = {}
        
        for split in ['train', 'val']:
            print(f"\n  {split}集时间戳:")
            self.report['timestamps'][split] = {}
            
            try:
                timestamps_raw = np.load(os.path.join(self.data_dir, f"{split}_timestamps{self.suffix}.npy"))
                
                # 转换时间戳
                if timestamps_raw.dtype == np.float64:
                    timestamps = pd.to_datetime(timestamps_raw, unit='s')
                else:
                    timestamps = pd.to_datetime(timestamps_raw)
                
                # 时间范围
                time_range = {
                    'start': str(timestamps.min()),
                    'end': str(timestamps.max()),
                    'duration_days': (timestamps.max() - timestamps.min()).days
                }
                
                print(f"    时间范围: {time_range['start']} 至 {time_range['end']}")
                print(f"    跨度: {time_range['duration_days']} 天")
                
                # 检查时间顺序
                is_sorted = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
                print(f"    时间顺序: {'✓ 已排序' if is_sorted else '✗ 未排序'}")
                
                # 检查重复
                n_unique = len(np.unique(timestamps))
                n_duplicates = len(timestamps) - n_unique
                print(f"    重复时间戳: {n_duplicates} 个")
                
                self.report['timestamps'][split] = {
                    'time_range': time_range,
                    'is_sorted': bool(is_sorted),
                    'n_unique': int(n_unique),
                    'n_duplicates': int(n_duplicates)
                }
                
            except Exception as e:
                print(f"    ❌ 检查失败: {e}")
                self.report['timestamps'][split]['error'] = str(e)
    
    def check_data_consistency(self):
        """检查数据一致性"""
        print("\n  检查训练集和验证集的一致性...")
        self.report['consistency'] = {}
        
        try:
            # 加载训练集和验证集的特征形状
            train_features = np.load(os.path.join(self.data_dir, f"train_features{self.suffix}.npy"), mmap_mode='r')
            val_features = np.load(os.path.join(self.data_dir, f"val_features{self.suffix}.npy"), mmap_mode='r')
            
            # 检查特征维度是否一致
            train_shape = train_features.shape[1:]  # 除了样本数之外的维度
            val_shape = val_features.shape[1:]
            
            shape_consistent = train_shape == val_shape
            print(f"    特征维度一致性: {'✓' if shape_consistent else '✗'}")
            
            if not shape_consistent:
                print(f"      训练集: {train_shape}")
                print(f"      验证集: {val_shape}")
            
            # 检查数据分布（采样）
            n_check = min(100, len(train_features), len(val_features))
            train_sample = train_features[np.random.choice(len(train_features), n_check)]
            val_sample = val_features[np.random.choice(len(val_features), n_check)]
            
            train_mean = np.mean(train_sample)
            val_mean = np.mean(val_sample)
            mean_diff = abs(train_mean - val_mean)
            
            print(f"    数据分布差异:")
            print(f"      训练集均值: {train_mean:.4f}")
            print(f"      验证集均值: {val_mean:.4f}")
            print(f"      差异: {mean_diff:.4f} {'✓ 正常' if mean_diff < 0.5 else '⚠️ 较大'}")
            
            self.report['consistency'] = {
                'shape_consistent': bool(shape_consistent),
                'train_shape': list(train_shape),
                'val_shape': list(val_shape),
                'train_mean': float(train_mean),
                'val_mean': float(val_mean),
                'mean_difference': float(mean_diff)
            }
            
        except Exception as e:
            print(f"    ❌ 检查失败: {e}")
            self.report['consistency']['error'] = str(e)
    
    def check_memory_usage(self):
        """检查内存使用情况"""
        self.report['memory_usage'] = {}
        total_size_gb = 0
        
        for split in ['train', 'val']:
            split_size = 0
            self.report['memory_usage'][split] = {}
            
            for file_type in ['features', 'stat_features', 'labels', 'timestamps']:
                filename = f"{split}_{file_type}{self.suffix}.npy"
                filepath = os.path.join(self.data_dir, filename)
                
                if os.path.exists(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    split_size += size_mb
                    self.report['memory_usage'][split][file_type] = float(size_mb)
            
            self.report['memory_usage'][split]['total_mb'] = float(split_size)
            total_size_gb += split_size / 1024
        
        self.report['memory_usage']['total_gb'] = float(total_size_gb)
        print(f"    总占用空间: {total_size_gb:.2f} GB")
    
    def generate_visualizations(self):
        """生成可视化图表"""
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'数据验证可视化 - {self.data_dir}', fontsize=16)
            
            # 1. 标签分布热力图
            ax = axes[0, 0]
            train_labels = np.load(os.path.join(self.data_dir, f"train_labels{self.suffix}.npy"))
            task_positive_rates = [np.mean(train_labels[:, i] > 0.5) for i in range(train_labels.shape[1])]
            
            # 重塑为3x4矩阵（3个时间窗口，4个震级范围）
            heatmap_data = np.array(task_positive_rates).reshape(3, 4)
            sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='YlOrRd', ax=ax)
            ax.set_title('任务正样本率热力图')
            ax.set_xticklabels(['M3.0-4.5', 'M4.5-5.5', 'M5.5-6.5', 'M6.5+'])
            ax.set_yticklabels(['7天', '14天', '30天'])
            
            # 2. 样本数量对比
            ax = axes[0, 1]
            if self.is_augmented and 'data_shapes' in self.report:
                train_samples = self.report['data_shapes']['train'].get('n_samples', 0)
                val_samples = self.report['data_shapes']['val'].get('n_samples', 0)
                
                ax.bar(['训练集', '验证集'], [train_samples, val_samples])
                ax.set_title('数据集样本数量')
                ax.set_ylabel('样本数')
                
                # 添加数值标签
                for i, v in enumerate([train_samples, val_samples]):
                    ax.text(i, v + v*0.01, str(v), ha='center')
            
            # 3. 特征值分布（采样）
            ax = axes[1, 0]
            features = np.load(os.path.join(self.data_dir, f"train_features{self.suffix}.npy"), mmap_mode='r')
            sample_features = features[np.random.choice(len(features), min(1000, len(features)))]
            ax.hist(sample_features.flatten(), bins=50, alpha=0.7, edgecolor='black')
            ax.set_title('特征值分布（采样）')
            ax.set_xlabel('特征值')
            ax.set_ylabel('频数')
            
            # 4. 时间分布
            ax = axes[1, 1]
            timestamps_raw = np.load(os.path.join(self.data_dir, f"train_timestamps{self.suffix}.npy"))
            if timestamps_raw.dtype == np.float64:
                timestamps = pd.to_datetime(timestamps_raw, unit='s')
            else:
                timestamps = pd.to_datetime(timestamps_raw)
            
            # 按月统计
            monthly_counts = pd.Series(timestamps).dt.to_period('M').value_counts().sort_index()
            monthly_counts.plot(kind='line', ax=ax, marker='o')
            ax.set_title('样本时间分布')
            ax.set_xlabel('月份')
            ax.set_ylabel('样本数')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图表
            viz_path = os.path.join(self.data_dir, f'validation_visualization{self.suffix}.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    ✓ 可视化已保存: {viz_path}")
            self.report['visualization_saved'] = True
            
        except Exception as e:
            print(f"    ❌ 生成可视化失败: {e}")
            self.report['visualization_error'] = str(e)

def compare_datasets(original_dir: str, augmented_dir: str):
    """比较原始数据和增强数据"""
    print("\n📊 比较原始数据和增强数据")
    print("="*60)
    
    # 加载增强配置
    config_path = os.path.join(augmented_dir, 'augmentation_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            aug_config = json.load(f)
        print(f"\n增强配置:")
        print(f"  增强因子: {aug_config.get('augmentation_factor', 'N/A')}")
        print(f"  最大正样本数: {aug_config.get('max_positive_samples', 'N/A')}")
    
    # 比较统计信息
    for split in ['train', 'val']:
        print(f"\n{split}集对比:")
        
        try:
            # 原始数据
            orig_labels = np.load(os.path.join(original_dir, f"{split}_labels.npy"))
            orig_features = np.load(os.path.join(original_dir, f"{split}_features.npy"), mmap_mode='r')
            
            # 增强数据
            aug_labels = np.load(os.path.join(augmented_dir, f"{split}_labels_aug.npy"))
            aug_features = np.load(os.path.join(augmented_dir, f"{split}_features_aug.npy"), mmap_mode='r')
            
            # 样本数对比
            print(f"  样本数: {len(orig_labels)} → {len(aug_labels)} "
                  f"(增加 {len(aug_labels) - len(orig_labels)})")
            
            # 正样本率对比
            orig_positive_rate = np.mean(np.any(orig_labels > 0.5, axis=1))
            aug_positive_rate = np.mean(np.any(aug_labels > 0.5, axis=1))
            print(f"  正样本率: {orig_positive_rate:.2%} → {aug_positive_rate:.2%} "
                  f"(+{(aug_positive_rate - orig_positive_rate):.2%})")
            
            # 文件大小对比
            orig_size = orig_features.nbytes / (1024**3)
            aug_size = aug_features.nbytes / (1024**3)
            print(f"  特征文件大小: {orig_size:.2f}GB → {aug_size:.2f}GB")
            
        except Exception as e:
            print(f"  ❌ 比较失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据验证工具')
    parser.add_argument('--data_dir', type=str, default='../../data/processed_grid',
                       help='数据目录')
    parser.add_argument('--augmented_dir', type=str, default='../../data/augmented_data',
                       help='增强数据目录')
    parser.add_argument('--check_augmented', action='store_true',
                       help='检查增强后的数据')
    parser.add_argument('--compare', action='store_true',
                       help='比较原始和增强数据')
    parser.add_argument('--no_viz', action='store_true',
                       help='不生成可视化')
    
    args = parser.parse_args()
    
    print("🔍 地震数据验证工具")
    print("="*60)
    
    if args.compare:
        # 比较模式
        compare_datasets(args.data_dir, args.augmented_dir)
    else:
        # 验证模式
        if args.check_augmented:
            data_dir = args.augmented_dir
            is_augmented = True
        else:
            data_dir = args.data_dir
            is_augmented = False
        
        validator = DataValidator(data_dir, is_augmented)
        
        # 如果不要可视化，临时修改方法
        if args.no_viz:
            validator.generate_visualizations = lambda: print("  跳过可视化生成")
        
        report = validator.validate_all()
        
        # 打印总结
        print("\n" + "="*60)
        print("📋 验证总结:")
        
        if report.get('all_files_exist'):
            print("  ✓ 所有文件都存在")
        else:
            print("  ✗ 缺少部分文件")
        
        if 'memory_usage' in report:
            print(f"  💾 总占用空间: {report['memory_usage']['total_gb']:.2f} GB")
        
        # 检查问题
        issues = []
        
        # 检查NaN/Inf
        for split in ['train', 'val']:
            if 'value_ranges' in report and split in report['value_ranges']:
                if report['value_ranges'][split].get('has_nan'):
                    issues.append(f"{split}集包含NaN值")
                if report['value_ranges'][split].get('has_inf'):
                    issues.append(f"{split}集包含Inf值")
        
        # 检查一致性
        if 'consistency' in report and not report['consistency'].get('shape_consistent', True):
            issues.append("训练集和验证集形状不一致")
        
        if issues:
            print("\n⚠️  发现的问题:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ 数据验证通过，未发现明显问题")

if __name__ == "__main__":
    main()