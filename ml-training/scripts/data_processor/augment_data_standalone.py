"""
独立的数据增强脚本
可以预先运行，生成增强后的数据集
"""

import numpy as np
import pandas as pd
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional
import argparse
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_aggregation import apply_data_augmentation_strategy

def check_data_statistics(data_dir: str) -> Dict:
    """检查数据统计信息"""
    print("\n📊 检查原始数据...")
    
    # 加载标签以计算正样本
    train_labels_path = os.path.join(data_dir, "train_labels.npy")
    train_labels = np.load(train_labels_path)
    
    # 详细分析每个任务的正样本
    print("\n📊 各任务正样本统计:")
    task_names = [
        "7d_M3-4.5", "7d_M4.5-5.5", "7d_M5.5-6.5", "7d_M6.5+",
        "14d_M3-4.5", "14d_M4.5-5.5", "14d_M5.5-6.5", "14d_M6.5+",
        "30d_M3-4.5", "30d_M4.5-5.5", "30d_M5.5-6.5", "30d_M6.5+"
    ]
    
    task_positive_counts = {}
    for task_idx in range(12):
        positive_count = np.sum(train_labels[:, task_idx] > 0.5)
        positive_ratio = positive_count / len(train_labels)
        print(f"  任务{task_idx:2d} ({task_names[task_idx]}): {positive_count:4d} ({positive_ratio:.2%})")
        task_positive_counts[f'task_{task_idx}'] = {
            'count': int(positive_count),
            'ratio': float(positive_ratio)
        }
    
    # 分析样本级别的正样本（任何任务为正）
    sample_has_positive = np.any(train_labels > 0.5, axis=1)
    positive_samples = np.sum(sample_has_positive)
    
    print(f"\n📊 样本级别统计:")
    print(f"  有正标签的样本数: {positive_samples} ({positive_samples/len(train_labels):.2%})")
    
    # 分析多任务正样本
    positive_task_counts_per_sample = np.sum(train_labels > 0.5, axis=1)
    for i in range(1, 13):
        count = np.sum(positive_task_counts_per_sample == i)
        if count > 0:
            print(f"  有{i}个正任务的样本: {count}")
    
    # 估算内存使用
    train_features = np.load(os.path.join(data_dir, "train_features.npy"), mmap_mode='r')
    sample_size_mb = train_features.nbytes / (1024 * 1024) / len(train_features)
    
    # 计算统计信息
    stats = {
        'total_samples': len(train_labels),
        'positive_samples': int(positive_samples),
        'positive_ratio': float(positive_samples / len(train_labels)),
        'task_positive_counts': task_positive_counts,
        'sample_size_mb': float(sample_size_mb)
    }
    
    return stats

def augment_dataset(data_dir: str, 
                   output_dir: str,
                   augmentation_factor: int = 10,
                   max_positive_samples: int = 1000,
                   validate: bool = True) -> None:
    """
    增强数据集并保存
    
    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        augmentation_factor: 增强倍数
        max_positive_samples: 最多增强的正样本数
        validate: 是否同时增强验证集
    """
    print(f"\n🚀 开始数据增强流程")
    print(f"  原始数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  增强倍数: {augmentation_factor}")
    print(f"  最大正样本数: {max_positive_samples}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config = {
        'augmentation_factor': augmentation_factor,
        'max_positive_samples': max_positive_samples,
        'timestamp': datetime.now().isoformat(),
        'source_dir': data_dir
    }
    
    with open(os.path.join(output_dir, 'augmentation_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 处理训练集
    print("\n📂 处理训练集...")
    process_split(
        data_dir=data_dir,
        output_dir=output_dir,
        split='train',
        augmentation_factor=augmentation_factor,
        max_positive_samples=max_positive_samples,
        apply_augmentation=True
    )
    
    # 处理验证集（不增强）
    if validate:
        print("\n📂 处理验证集...")
        process_split(
            data_dir=data_dir,
            output_dir=output_dir,
            split='val',
            augmentation_factor=1,
            max_positive_samples=None,
            apply_augmentation=False
        )
    
    print("\n✅ 数据增强完成！")
    print(f"  增强后的数据保存在: {output_dir}")

def process_split(data_dir: str,
                 output_dir: str,
                 split: str,
                 augmentation_factor: int,
                 max_positive_samples: Optional[int],
                 apply_augmentation: bool) -> None:
    """处理单个数据集分割"""
    
    # 文件路径
    features_path = os.path.join(data_dir, f"{split}_features.npy")
    stat_features_path = os.path.join(data_dir, f"{split}_stat_features.npy")
    labels_path = os.path.join(data_dir, f"{split}_labels.npy")
    timestamps_path = os.path.join(data_dir, f"{split}_timestamps.npy")
    
    # 检查文件是否存在
    required_files = [features_path, stat_features_path, labels_path, timestamps_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return
    
    # 加载数据
    print(f"  加载 {split} 数据...")
    start_time = time.time()
    
    features = np.load(features_path)
    stat_features = np.load(stat_features_path)
    labels = np.load(labels_path)
    timestamps_raw = np.load(timestamps_path)
    
    print(f"  ✓ 加载完成 ({time.time() - start_time:.1f}秒)")
    print(f"  原始形状: features={features.shape}, labels={labels.shape}")
    
    original_len = len(features)
    
    # 应用数据增强
    if apply_augmentation:
        print(f"\n  应用数据增强...")
        
        # 使用改进的数据增强方法 - 修复参数名称
        features_aug, labels_aug = apply_data_augmentation_strategy(
            features, 
            labels, 
            augmentation_factor=augmentation_factor,
            probability_threshold=0.2,  # 基于概率值而非二值化
            max_augment_samples=max_positive_samples,  # 修正参数名
            preserve_negatives=True  # 确保保留负样本
        )
        
        # 同步增强统计特征和时间戳
        print(f"  同步增强统计特征...")
        num_augmented = len(features_aug) - len(features)
        
        if num_augmented > 0:
            # 找出被增强的原始样本索引（基于平均概率而非二值化）
            mean_probs = np.mean(labels, axis=1)
            high_prob_mask = mean_probs > 0.2  # 使用与增强相同的阈值
            high_prob_indices = np.where(high_prob_mask)[0]
            
            if max_positive_samples is not None and len(high_prob_indices) > max_positive_samples:
                # 按概率排序，选择概率最高的样本
                sorted_indices = high_prob_indices[np.argsort(mean_probs[high_prob_indices])[::-1]]
                high_prob_indices = sorted_indices[:max_positive_samples]
            
            # 生成增强的统计特征
            augmented_stat_features = []
            
            for idx in high_prob_indices:
                base_stat = stat_features[idx]
                for _ in range(augmentation_factor):
                    if len(augmented_stat_features) >= num_augmented:
                        break
                    noise = np.random.normal(0, 0.03, base_stat.shape)
                    aug_stat = base_stat * (1 + noise)
                    aug_stat = np.maximum(aug_stat, 0)
                    augmented_stat_features.append(aug_stat)
            
            augmented_stat_features = np.array(augmented_stat_features[:num_augmented])
            stat_features_aug = np.concatenate([stat_features, augmented_stat_features], axis=0)
            
            # 同步时间戳
            print(f"  同步时间戳...")
            if timestamps_raw.dtype == np.float64:
                timestamps = pd.to_datetime(timestamps_raw, unit='s')
            else:
                timestamps = pd.to_datetime(timestamps_raw)
            
            # 为增强的样本复制时间戳
            augmented_timestamps = []
            for idx in high_prob_indices:
                for _ in range(augmentation_factor):
                    if len(augmented_timestamps) >= num_augmented:
                        break
                    augmented_timestamps.append(timestamps[idx])
            
            # 合并时间戳
            timestamps_aug = pd.DatetimeIndex(list(timestamps) + augmented_timestamps[:num_augmented])
            
            if timestamps_raw.dtype == np.float64:
                timestamps_aug_raw = timestamps_aug.astype(np.int64) / 1e9
            else:
                timestamps_aug_raw = timestamps_aug.values
        else:
            stat_features_aug = stat_features
            timestamps_aug_raw = timestamps_raw
    else:
        # 不增强，直接复制
        features_aug = features
        stat_features_aug = stat_features
        labels_aug = labels
        timestamps_aug_raw = timestamps_raw
    
    # 保存增强后的数据
    print(f"\n  保存增强后的数据...")
    
    output_features_path = os.path.join(output_dir, f"{split}_features_aug.npy")
    output_stat_features_path = os.path.join(output_dir, f"{split}_stat_features_aug.npy")
    output_labels_path = os.path.join(output_dir, f"{split}_labels_aug.npy")
    output_timestamps_path = os.path.join(output_dir, f"{split}_timestamps_aug.npy")
    
    np.save(output_features_path, features_aug)
    np.save(output_stat_features_path, stat_features_aug)
    np.save(output_labels_path, labels_aug)
    np.save(output_timestamps_path, timestamps_aug_raw)
    
    # 保存统计信息
    # 计算基于概率值的统计（而非二值化）
    mean_probs_before = np.mean(labels)
    mean_probs_after = np.mean(labels_aug)
    high_prob_before = np.mean(np.mean(labels, axis=1) > 0.2)
    high_prob_after = np.mean(np.mean(labels_aug, axis=1) > 0.2)
    
    stats = {
        'split': split,
        'original_samples': original_len,
        'augmented_samples': len(features_aug),
        'augmentation_ratio': len(features_aug) / original_len,
        'mean_probability_before': float(mean_probs_before),
        'mean_probability_after': float(mean_probs_after),
        'high_prob_ratio_before': float(high_prob_before),
        'high_prob_ratio_after': float(high_prob_after),
        'file_sizes_mb': {
            'features': os.path.getsize(output_features_path) / (1024 * 1024),
            'stat_features': os.path.getsize(output_stat_features_path) / (1024 * 1024),
            'labels': os.path.getsize(output_labels_path) / (1024 * 1024),
            'timestamps': os.path.getsize(output_timestamps_path) / (1024 * 1024)
        }
    }
    
    stats_path = os.path.join(output_dir, f"{split}_augmentation_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  ✓ 保存完成")
    print(f"  最终形状: features={features_aug.shape}, labels={labels_aug.shape}")
    print(f"  平均概率: {stats['mean_probability_before']:.4f} → {stats['mean_probability_after']:.4f}")
    print(f"  高概率样本比例: {stats['high_prob_ratio_before']:.2%} → {stats['high_prob_ratio_after']:.2%}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='独立数据增强工具')
    parser.add_argument('--data_dir', type=str, default='../../data/processed_grid',
                       help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='../../data/augmented_data',
                       help='输出目录')
    parser.add_argument('--augment_factor', type=int, default=6,
                       help='增强倍数')
    parser.add_argument('--max_positive', type=int, default=1000,
                       help='最大正样本数量')
    parser.add_argument('--check_only', action='store_true',
                       help='仅检查数据统计信息')
    parser.add_argument('--no_validate', action='store_true',
                       help='不处理验证集')
    
    args = parser.parse_args()
    
    print("🌍 地震数据增强工具")
    print("="*60)
    
    # 检查数据统计
    stats = check_data_statistics(args.data_dir)
    
    print(f"\n📊 数据统计:")
    print(f"  总样本数: {stats['total_samples']:,}")
    print(f"  正样本数: {stats['positive_samples']:,} ({stats['positive_ratio']:.2%})")
    print(f"  每个样本大小: {stats['sample_size_mb']:.2f} MB")
    
    # 估算增强后的大小
    estimated_positive = min(stats['positive_samples'], args.max_positive)
    estimated_total = stats['total_samples'] + estimated_positive * args.augment_factor
    estimated_size_gb = estimated_total * stats['sample_size_mb'] / 1024
    
    print(f"\n📈 增强预估:")
    print(f"  将增强正样本数: {estimated_positive:,}")
    print(f"  新增样本数: {estimated_positive * args.augment_factor:,}")
    print(f"  增强后总样本数: {estimated_total:,}")
    print(f"  预计占用空间: {estimated_size_gb:.1f} GB")
    
    if args.check_only:
        print("\n仅检查模式，退出。")
        return
    
    # 确认继续
    print(f"\n⚠️  注意：增强后的数据将占用约 {estimated_size_gb:.1f} GB 空间")
    confirm = input("是否继续? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("操作已取消")
        return
    
    # 执行增强
    start_time = time.time()
    
    augment_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        augmentation_factor=args.augment_factor,
        max_positive_samples=args.max_positive,
        validate=not args.no_validate
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  总耗时: {elapsed_time/60:.1f} 分钟")
    
    # 显示如何在训练中使用
    print("\n💡 使用提示:")
    print(f"  在训练时，将 data_dir 设置为: {args.output_dir}")
    print(f"  并将文件名从 'xxx.npy' 改为 'xxx_aug.npy'")
    print(f"  或者在 integrated_training_pipeline.py 中设置:")
    print(f"    config['data_dir'] = '{args.output_dir}'")
    print(f"    config['data_type'] = 'augmented'")

if __name__ == "__main__":
    main()