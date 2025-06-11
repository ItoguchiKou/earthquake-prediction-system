"""
解耦的地震预测训练流程
数据增强已完全分离，只使用预处理好的数据
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# 导入必要的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lightweight_model import create_lightweight_model
from models.loss_functions import EarthquakeRegressionLoss
from models.model_utils import save_checkpoint, load_checkpoint

class EarthquakeDataset(Dataset):
    """简化的地震数据集 - 不包含任何增强逻辑"""
    
    def __init__(self,
                 features_path: str,
                 stat_features_path: str,
                 labels_path: str,
                 timestamps_path: str,
                 use_mmap: bool = True):
        """
        初始化数据集
        
        Args:
            features_path: 特征文件路径
            stat_features_path: 统计特征文件路径
            labels_path: 标签文件路径
            timestamps_path: 时间戳文件路径
            use_mmap: 是否使用内存映射
        """
        # 加载数据
        if use_mmap:
            try:
                self.features = np.load(features_path, mmap_mode='r')
                self.stat_features = np.load(stat_features_path, mmap_mode='r')
                self.labels = np.load(labels_path, mmap_mode='r')
            except:
                self.features = np.load(features_path)
                self.stat_features = np.load(stat_features_path)
                self.labels = np.load(labels_path)
        else:
            self.features = np.load(features_path)
            self.stat_features = np.load(stat_features_path)
            self.labels = np.load(labels_path)
        
        # 加载时间戳
        timestamps_raw = np.load(timestamps_path)
        if timestamps_raw.dtype == np.float64:
            self.timestamps = pd.to_datetime(timestamps_raw, unit='s')
        else:
            self.timestamps = pd.to_datetime(timestamps_raw)
        
        # 数据信息
        self.grid_height = self.features.shape[2]
        self.grid_width = self.features.shape[3]
        
        # 计算正样本率（这里改为高概率样本率）
        high_prob_mask = np.any(self.labels > 0.3, axis=1)
        high_prob_ratio = np.mean(high_prob_mask)
        
        print(f"✅ 数据集加载完成:")
        print(f"  样本数: {len(self.features)}")
        print(f"  特征形状: {self.features.shape}")
        print(f"  高概率(>0.3)样本率: {high_prob_ratio:.2%}")
    
    def get_sampler(self):
        """获取采样器（兼容性方法）"""
        # 对于解耦的管线，不使用特殊采样器
        # 让DataLoader使用默认的随机采样
        return None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # 确保返回连续的数组
        if isinstance(self.features, np.memmap):
            features = np.array(self.features[idx], dtype=np.float32)
            stat_features = np.array(self.stat_features[idx], dtype=np.float32)
            labels = np.array(self.labels[idx], dtype=np.float32)
        else:
            features = self.features[idx].astype(np.float32)
            stat_features = self.stat_features[idx].astype(np.float32)
            labels = self.labels[idx].astype(np.float32)
        
        return (torch.from_numpy(features).float(),
                torch.from_numpy(stat_features).float(),
                torch.from_numpy(labels).float())

class SimplifiedTrainingPipeline:
    """简化的训练流程 - 专注于训练本身"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练流程
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        self._set_random_seed(config.get('random_seed', 42))
        
        # 创建输出目录
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(self.save_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # 有效任务（不再排除任何任务）
        self.valid_tasks = config.get('valid_tasks', list(range(12)))
        
        print(f"🚀 训练流程初始化")
        print(f"  设备: {self.device}")
        print(f"  输出目录: {self.save_dir}")
        print(f"  有效任务: {self.valid_tasks}")
    
    def _set_random_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def prepare_data(self) -> Tuple[Dataset, Dataset]:
        """准备数据集"""
        print("\n📂 准备数据集...")
        
        data_dir = self.config['data_dir']
        data_type = self.config.get('data_type', 'augmented')  # 'original' or 'augmented'
        
        # 根据数据类型选择文件后缀
        suffix = '_aug' if data_type == 'augmented' else ''
        
        print(f"  数据类型: {data_type}")
        print(f"  数据目录: {data_dir}")
        
        # 创建训练集
        train_dataset = EarthquakeDataset(
            features_path=os.path.join(data_dir, f"train_features{suffix}.npy"),
            stat_features_path=os.path.join(data_dir, f"train_stat_features{suffix}.npy"),
            labels_path=os.path.join(data_dir, f"train_labels{suffix}.npy"),
            timestamps_path=os.path.join(data_dir, f"train_timestamps{suffix}.npy"),
            use_mmap=self.config.get('use_mmap', True)
        )
        
        # 创建验证集
        val_dataset = EarthquakeDataset(
            features_path=os.path.join(data_dir, f"val_features{suffix}.npy"),
            stat_features_path=os.path.join(data_dir, f"val_stat_features{suffix}.npy"),
            labels_path=os.path.join(data_dir, f"val_labels{suffix}.npy"),
            timestamps_path=os.path.join(data_dir, f"val_timestamps{suffix}.npy"),
            use_mmap=self.config.get('use_mmap', True)
        )
        
        return train_dataset, val_dataset
    
    def create_model(self, data_shape: Tuple[int, ...]) -> nn.Module:
        """创建模型"""
        print("\n🏗️ 创建模型...")
        
        model_type = self.config.get('model_type', 'standard')
        
        # 创建轻量级模型
        model = create_lightweight_model(
            data_shape=data_shape,
            model_type=model_type,
            valid_tasks=self.valid_tasks
        )
        
        # 设置损失函数 - 修复参数
        model.loss_function = EarthquakeRegressionLoss(
            loss_type=self.config.get('loss_type', 'huber'),
            consistency_weight=self.config.get('consistency_weight', 0.1),
            ignore_tasks=self.config.get('ignore_tasks', [])
        ).to(self.device)
        
        # 添加compute_loss方法
        model.compute_loss = lambda pred, target: model.loss_function(pred, target)
        
        return model.to(self.device)
    
    def create_optimizer_scheduler(self, model: nn.Module, num_training_steps: int):
        """创建优化器和调度器"""
        print("\n⚙️ 设置优化器和调度器...")
        
        # 优化器
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"未知优化器: {optimizer_type}")
        
        # 调度器
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['epochs'],
                eta_min=lr * 0.01
            )
        elif scheduler_type == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=num_training_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',  # 对于回归任务，最小化损失
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None
        
        return optimizer, scheduler
    
    def train(self):
        """执行训练"""
        print("\n🎯 开始训练流程...")
        
        # 准备数据
        train_dataset, val_dataset = self.prepare_data()
        
        # 创建数据加载器
        batch_size = self.config.get('batch_size', 16)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=torch.cuda.is_available() and not isinstance(train_dataset.features, np.memmap)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=torch.cuda.is_available() and not isinstance(val_dataset.features, np.memmap)
        )
        
        # 创建模型
        sample_shape = (batch_size,) + train_dataset.features.shape[1:]
        model = self.create_model(sample_shape)
        
        # 计算总训练步数
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        num_training_steps = len(train_loader) * self.config['epochs'] // gradient_accumulation_steps
        
        # 创建优化器和调度器
        optimizer, scheduler = self.create_optimizer_scheduler(model, num_training_steps)
        
        # 导入训练器
        from training_strategy import ImprovedEarthquakeTrainer
        
        # 创建训练器
        trainer = ImprovedEarthquakeTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=self.config
        )
        
        # 执行训练
        trainer.train()
        
        return trainer

def create_training_config(mode: str = 'standard') -> Dict[str, Any]:
    """创建训练配置 - 回归版本"""
    
    base_config = {
        'data_dir': '../data/augmented_data',
        'data_type': 'augmented',
        'save_dir': f'earthquake_regression_model_{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'random_seed': 42,
        'ignore_tasks': [],  # 不再忽略任何任务
        'valid_tasks': list(range(12)),  # 所有任务都有效
        'loss_type': 'huber',  # 使用Huber损失
        'consistency_weight': 0.1,
        'num_workers': 0,
        'use_mmap': True,
        'grad_clip': 1.0,
        'save_interval': 5,
        'optimizer': 'adam',
        'weight_decay': 1e-4
    }
    
    # 根据模式调整配置
    configs = {
        'debug': {
            **base_config,
            'model_type': 'minimal',
            'batch_size': 8,
            'learning_rate': 1e-3,
            'epochs': 5,
            'gradient_accumulation_steps': 1,
            'early_stop_patience': 3,
            'scheduler': 'cosine'
        },
        'standard': {
            **base_config,
            'model_type': 'standard',
            'batch_size': 16,
            'learning_rate': 5e-4,
            'epochs': 50,
            'gradient_accumulation_steps': 2,
            'early_stop_patience': 10,
            'scheduler': 'cosine'
        },
        'fast': {
            **base_config,
            'model_type': 'minimal',
            'batch_size': 32,
            'learning_rate': 1e-3,
            'epochs': 20,
            'gradient_accumulation_steps': 1,
            'early_stop_patience': 5,
            'scheduler': 'onecycle'
        },
        'intensive': {
            **base_config,
            'model_type': 'standard',
            'batch_size': 8,
            'learning_rate': 3e-4,
            'epochs': 80,
            'gradient_accumulation_steps': 4,
            'early_stop_patience': 15,
            'scheduler': 'plateau',
            'weight_decay': 5e-4
        },
        'aggressive': {
            **base_config,
            'model_type': 'standard',
            'batch_size': 16,
            'learning_rate': 2e-3,  # 从5e-4提高到2e-3
            'epochs': 50,
            'gradient_accumulation_steps': 1,  # 从2改为1
            'early_stop_patience': 10,
            'scheduler': 'onecycle',  # 从cosine改为onecycle
            'loss_type': 'mse',  # 从huber改为mse
            'consistency_weight': 0.05  # 从0.1降低到0.05
        }
    }
    
    if mode not in configs:
        raise ValueError(f"未知模式: {mode}. 可选: {list(configs.keys())}")
    
    return configs[mode]

def main():
    """主函数"""
    print("🌍 地震预测训练系统")
    print("="*60)
    
    # 检查数据
    data_dirs = {
        'original': '../data/processed_grid',
        'augmented': '../data/augmented_data'
    }
    
    print("\n📊 检查可用数据:")
    available_data = {}
    
    for data_type, data_dir in data_dirs.items():
        if data_type == 'augmented':
            train_file = os.path.join(data_dir, 'train_features_aug.npy')
        else:
            train_file = os.path.join(data_dir, 'train_features.npy')
        
        if os.path.exists(train_file):
            # 快速检查样本数
            try:
                features = np.load(train_file, mmap_mode='r')
                n_samples = len(features)
                available_data[data_type] = n_samples
                print(f"  ✓ {data_type}: {n_samples} 个训练样本")
            except:
                print(f"  ✗ {data_type}: 无法读取")
        else:
            print(f"  ✗ {data_type}: 不存在")
    
    # 选择数据类型
    if len(available_data) == 0:
        print("\n❌ 没有找到可用的数据!")
        print("请先运行数据处理脚本。")
        return
    
    if len(available_data) == 1:
        data_type = list(available_data.keys())[0]
        print(f"\n自动选择: {data_type} 数据")
    else:
        print("\n请选择数据类型:")
        print("1. 原始数据")
        print("2. 增强数据 (推荐)")
        
        choice = input("请输入选择 (1/2, 默认为2): ").strip() or "2"
        data_type = 'original' if choice == '1' else 'augmented'
    
    # 选择训练模式
    print("\n请选择训练模式:")
    print("1. 调试模式 (5 epochs, 快速测试)")
    print("2. 标准模式 (50 epochs, 推荐)")
    print("3. 快速模式 (20 epochs, 快速训练)")
    print("4. 深度模式 (80 epochs, 最佳效果)")
    print("5. 激进模式 (30 epochs, MSE损失)")
    mode_input = input("请输入选择 (1/2/3/4/5, 默认为2): ").strip() or "2"
    mode_map = {'1': 'debug', '2': 'standard', '3': 'fast', '4': 'intensive', '5': 'aggressive'}
    mode = mode_map.get(mode_input, 'standard')
    
    # 创建配置
    config = create_training_config(mode)
    config['data_dir'] = data_dirs[data_type]
    config['data_type'] = data_type
    
    print(f"\n📋 训练配置 ({mode}模式):")
    print(f"  数据: {data_type} ({available_data.get(data_type, 0)} 样本)")
    print(f"  模型: {config['model_type']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  训练轮数: {config['epochs']}")
    print(f"  损失函数: {config['loss_type']}")
    print(f"  调度器: {config.get('scheduler', 'cosine')}")
    
    # 确认开始训练
    confirm = input("\n是否开始训练? (y/n): ").strip().lower()
    if confirm != 'y':
        print("训练已取消")
        return
    
    # 创建训练流程
    pipeline = SimplifiedTrainingPipeline(config)
    
    try:
        # 执行训练
        trainer = pipeline.train()
        print("\n🎉 训练完成!")
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()