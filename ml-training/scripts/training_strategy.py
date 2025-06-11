"""
地震预测模型训练策略 - 回归版本
针对概率预测任务优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import torch.profiler
import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# 导入模型模块
from models.model_architecture import EarthquakePredictionModel
from models.model_utils import save_checkpoint, load_checkpoint, AttentionVisualizer
# 导入改进的损失函数
from models.loss_functions import EarthquakeRegressionLoss

# GPU内存监控器（保持不变）
class GPUMemoryMonitor:
    """GPU内存监控器"""
    
    @staticmethod
    def log_memory_usage(stage: str = "", device=None):
        if torch.cuda.is_available():
            if device is None:
                device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            
            print(f"📊 GPU内存 {stage}:")
            print(f"   已分配: {allocated:.2f}GB")
            print(f"   已保留: {reserved:.2f}GB") 
            print(f"   峰值: {max_allocated:.2f}GB")
    
    @staticmethod
    def optimize_memory():
        """优化GPU内存使用"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class EarthquakeDataset(Dataset):
    """地震数据集类 - 保持不变，但修改权重计算"""
    
    def __init__(self, 
                 features_path: str,
                 stat_features_path: str,
                 labels_path: str,
                 timestamps_path: str,
                 sample_weights_path: str = None,
                 augment: bool = False,
                 balance_sampling: bool = True,
                 use_mmap: bool = True):
        """初始化数据集"""
        # 使用内存映射模式加载大文件
        if use_mmap:
            try:
                self.features = np.load(features_path, mmap_mode='r')
                self.stat_features = np.load(stat_features_path, mmap_mode='r')
                self.labels = np.load(labels_path, mmap_mode='r')
                print("  ✓ 使用内存映射模式加载数据")
            except Exception as e:
                print(f"  ⚠️ 内存映射失败，使用常规加载: {e}")
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
            
        self.augment = augment
        self.balance_sampling = balance_sampling
        self.use_mmap = use_mmap
        
        # 获取实际的网格尺寸
        self.grid_height = self.features.shape[2]
        self.grid_width = self.features.shape[3]
        
        # 确保是10×8的维度
        if self.grid_height != 10 or self.grid_width != 8:
            print(f"⚠️ 警告：数据维度不是10×8，而是{self.grid_height}×{self.grid_width}")
            print("   这可能是旧版本的数据，需要重新处理！")
        
        print(f"📊 数据集加载完成:")
        print(f"  时序特征形状: {self.features.shape}")
        print(f"  统计特征形状: {self.stat_features.shape}")
        print(f"  标签形状: {self.labels.shape}")
        print(f"  样本数量: {len(self.features)}")
        print(f"  网格尺寸: {self.grid_height} × {self.grid_width}")
        
        # 改进的样本权重计算 - 基于概率值而非二值化
        if sample_weights_path and os.path.exists(sample_weights_path):
            if use_mmap:
                try:
                    self.sample_weights = np.load(sample_weights_path, mmap_mode='r')
                    print("  ✓ 使用预计算的样本权重")
                except:
                    self.sample_weights = np.load(sample_weights_path)
            else:
                self.sample_weights = np.load(sample_weights_path)
        elif balance_sampling:
            print("  ⚠️ 未找到预计算权重，重新计算...")
            self.sample_weights = self._calculate_regression_sample_weights()
        else:
            self.sample_weights = None
    
    def _calculate_regression_sample_weights(self) -> np.ndarray:
        """回归任务的样本权重计算 - 基于概率值"""
        sample_weights = np.ones(len(self.labels))
        
        # 有效任务（所有任务）
        valid_tasks = list(range(12))
        
        for i in range(len(self.labels)):
            sample_labels = self.labels[i]
            weight = 1.0
            
            # 基于平均概率值计算权重
            mean_prob = np.mean(sample_labels)
            
            # 对高概率样本给予更高权重
            if mean_prob > 0.3:
                weight *= 2.0
            elif mean_prob > 0.2:
                weight *= 1.5
            
            # 对大震任务的高概率给予额外权重
            for task_idx in [3, 7, 11]:  # M6.5+的任务
                if sample_labels[task_idx] > 0.3:
                    weight *= 2.0
            
            sample_weights[i] = weight
        
        # 归一化权重
        sample_weights = sample_weights / sample_weights.mean()
        sample_weights = np.clip(sample_weights, 0.1, 10.0)
        
        return sample_weights
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # 读取数据
        if isinstance(self.features, np.memmap):
            features = np.array(self.features[idx], dtype=np.float32)
            stat_features = np.array(self.stat_features[idx], dtype=np.float32)
            labels = np.array(self.labels[idx], dtype=np.float32)
        else:
            features = self.features[idx].astype(np.float32)
            stat_features = self.stat_features[idx].astype(np.float32)
            labels = self.labels[idx].astype(np.float32)
        
        # 数据增强
        if self.augment:
            features = self._augment_data(features)
        
        # 确保数据连续
        features = np.ascontiguousarray(features, dtype=np.float32)
        stat_features = np.ascontiguousarray(stat_features, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.float32)
        
        return (torch.from_numpy(features).float(), 
                torch.from_numpy(stat_features).float(), 
                torch.from_numpy(labels).float())
    
    def _augment_data(self, features: np.ndarray) -> np.ndarray:
        """数据增强 - 适合回归任务"""
        features = features.astype(np.float32)
        
        # 时间扰动
        if np.random.random() < 0.3:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                features = np.roll(features, shift, axis=0)
        
        # 轻微的特征噪声
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, features.shape)
            features = np.clip(features + noise, 0, 1)
        
        return features.astype(np.float32)
    
    def get_sampler(self):
        """获取加权采样器"""
        if self.sample_weights is not None:
            return WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
        return None

class ImprovedEarthquakeTrainer:
    """改进的地震预测模型训练器 - 回归版本"""
    
    def __init__(self,
                 model: EarthquakePredictionModel,
                 train_dataset: EarthquakeDataset,
                 val_dataset: EarthquakeDataset,
                 config: Dict[str, Any]):
        """初始化训练器"""
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 检查模型输入尺寸是否与数据匹配
        expected_height = train_dataset.grid_height
        expected_width = train_dataset.grid_width
        # 对于不规则网格，期望是10×8
        if expected_height != 10 or expected_width != 8:
            print(f"⚠️ 警告: 数据维度({expected_height}×{expected_width})不是期望的10×8")
            print("   请确保使用新的网格系统处理数据！")
    
        model_height = model.input_height if hasattr(model, 'input_height') else None
        model_width = model.input_width if hasattr(model, 'input_width') else None
        
        if model_height != expected_height or model_width != expected_width:
            print(f"⚠️ 警告: 模型输入尺寸({model_height}×{model_width})与"
                f"数据尺寸({expected_height}×{expected_width})不匹配")
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 替换模型的损失函数为回归版本 - 修复参数
        self.model.loss_function = EarthquakeRegressionLoss(
            loss_type=config.get('loss_type', 'huber'),
            consistency_weight=config.get('consistency_weight', 0.1),
            ignore_tasks=config.get('ignore_tasks', [])
        ).to(self.device)
        
        # 初始化混合精度训练
        self.use_amp = torch.cuda.is_available() and config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 创建数据加载器
        self._create_data_loaders()
        
        # 创建优化器和调度器
        self._create_optimizer_scheduler()
        
        # 初始化训练状态
        self.epoch = 0
        self.best_score = float('inf')  # 对于回归，越小越好
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        self.val_metrics_history = []  # 保存详细指标历史
        
        # 创建保存目录
        self.save_dir = config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 有效任务列表（不再排除任何任务）
        self.valid_tasks = config.get('valid_tasks', list(range(12)))
        
        print(f"🚀 回归训练器初始化完成:")
        print(f"  设备: {self.device}")
        print(f"  训练样本: {len(train_dataset)}")
        print(f"  验证样本: {len(val_dataset)}")
        print(f"  批次大小: {config['batch_size']}")
        print(f"  损失类型: {config.get('loss_type', 'huber')}")
        print(f"  有效任务: {self.valid_tasks}")
        print(f"  学习率: {config['learning_rate']}")
        print(f"  梯度累积步数: {config.get('gradient_accumulation_steps', 1)}")
        print(f"  混合精度训练: {self.use_amp}")
        
        if torch.cuda.is_available():
            GPUMemoryMonitor.log_memory_usage("初始化完成", self.device)
    
    def _create_data_loaders(self):
        """创建数据加载器"""
        import platform
        is_windows = platform.system() == 'Windows'
        
        if is_windows:
            num_workers = 0
            print("  ⚠️ Windows系统检测到，使用单线程数据加载")
        else:
            num_workers = self.config.get('num_workers', 2)
        
        persistent_workers = True if num_workers > 0 else False
        pin_memory = torch.cuda.is_available() and self.config.get('pin_memory', True)
        
        # 训练数据加载器
        train_sampler = self.train_dataset.get_sampler()
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        # 验证数据加载器
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else None
        )
    
    def _create_optimizer_scheduler(self):
        """创建优化器和学习率调度器"""
        base_lr = self.config['learning_rate']
        
        # 分层学习率
        param_groups = []
        
        # 检查模型的各个组件是否存在
        if hasattr(self.model, 'spatial_encoder'):
            param_groups.append({
                'params': self.model.spatial_encoder.parameters(),
                'lr': base_lr * 1.0,
                'name': 'spatial_encoder'
            })
        
        if hasattr(self.model, 'temporal_encoder'):
            param_groups.append({
                'params': self.model.temporal_encoder.parameters(),
                'lr': base_lr * 0.8,
                'name': 'temporal_encoder'
            })
        
        if hasattr(self.model, 'attention_fusion'):
            param_groups.append({
                'params': self.model.attention_fusion.parameters(),
                'lr': base_lr * 0.6,
                'name': 'attention_fusion'
            })
        
        if hasattr(self.model, 'multi_task_head'):
            param_groups.append({
                'params': self.model.multi_task_head.parameters(),
                'lr': base_lr * 1.2,
                'name': 'multi_task_head'
            })
        
        if hasattr(self.model, 'task_head'):
            param_groups.append({
                'params': self.model.task_head.parameters(),
                'lr': base_lr * 1.2,
                'name': 'task_head'
            })
        
        # 损失函数参数
        if hasattr(self.model.loss_function, 'parameters'):
            param_groups.append({
                'params': self.model.loss_function.parameters(),
                'lr': base_lr * 0.5,
                'name': 'loss_function'
            })
        
        # 过滤空参数组
        param_groups = [g for g in param_groups if len(list(g['params'])) > 0]
        
        # 如果没有分组，使用所有参数
        if not param_groups:
            param_groups = self.model.parameters()
        
        # 创建优化器
        if self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=self.config.get('weight_decay', 1e-4),
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                param_groups,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config['optimizer']}")
        
        # 创建学习率调度器
        if self.config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=base_lr * 0.01
            )
        elif self.config['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',  # 对于回归任务，最小化损失
                factor=0.5,
                patience=self.config.get('scheduler_patience', 5),
                verbose=True
            )
        elif self.config['scheduler'] == 'warmup_cosine':
            self.scheduler = self._create_warmup_cosine_scheduler()
        elif self.config['scheduler'] == 'onecycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=base_lr * 10,  # 峰值学习率
                epochs=self.config['epochs'],
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3,  # 30%用于上升
                anneal_strategy='cos'
            )
        else:
            self.scheduler = None
    
    def _create_warmup_cosine_scheduler(self):
        """创建预热+余弦退火调度器"""
        warmup_epochs = self.config.get('warmup_epochs', 5)
        total_epochs = self.config['epochs']
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        num_batches = 0
        
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        empty_cache_freq = self.config.get('empty_cache_freq', 5)
        
        epoch_start_time = time.time()
        batch_times = []
        
        print(f"  开始训练，共 {len(self.train_loader)} 个批次...")
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (features, stat_features, labels) in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            if batch_idx % empty_cache_freq == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 移动数据到设备
            features = features.to(self.device, non_blocking=True).float()
            stat_features = stat_features.to(self.device, non_blocking=True).float()
            labels = labels.to(self.device, non_blocking=True).float()
            
            # 前向传播
            with autocast(enabled=self.use_amp):
                # 检查模型是否支持统计特征
                if hasattr(self.model, 'forward_with_stat_features'):
                    predictions = self.model.forward_with_stat_features(features, stat_features)
                else:
                    predictions = self.model(features)
            
            # 计算损失
            with autocast(enabled=False):
                loss_info = self._compute_loss_safe(predictions, labels)
                total_loss_batch = loss_info['total_loss'] / accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(total_loss_batch).backward()
            else:
                total_loss_batch.backward()
            
            # 梯度累积和优化
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 这里是梯度裁剪发生的地方
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
                
                # 如果使用OneCycle调度器，每步更新
                if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
            else:
                grad_norm = 0.0
            
            # 累积损失和指标
            total_loss += total_loss_batch.item() * accumulation_steps
            total_mae += loss_info.get('mae', 0.0)
            total_rmse += loss_info.get('rmse', 0.0)
            num_batches += 1
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # 进度显示
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == len(self.train_loader) - 1:
                avg_batch_time = np.mean(batch_times)
                eta_minutes = avg_batch_time * (len(self.train_loader) - batch_idx - 1) / 60
                
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)} "
                    f"| Loss: {total_loss_batch.item() * accumulation_steps:.4f} "
                    f"| MAE: {loss_info.get('mae', 0.0):.4f} "
                    f"| 批次耗时: {batch_time:.2f}s "
                    f"| ETA: {eta_minutes:.1f}分钟")
        
        # 平均损失
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_rmse = total_rmse / num_batches
        
        epoch_time = time.time() - epoch_start_time
        print(f"\n  训练阶段完成: 耗时 {epoch_time:.1f}秒")
        
        return {
            'total_loss': avg_loss,
            'mae': avg_mae,
            'rmse': avg_rmse
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch - 回归评估指标"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for features, stat_features, labels in self.val_loader:
                features = features.to(self.device, non_blocking=True).float()
                stat_features = stat_features.to(self.device, non_blocking=True).float()
                labels = labels.to(self.device, non_blocking=True).float()
                
                with autocast(enabled=self.use_amp):
                    if hasattr(self.model, 'forward_with_stat_features'):
                        predictions = self.model.forward_with_stat_features(features, stat_features)
                    else:
                        predictions = self.model(features)
                
                with autocast(enabled=False):
                    loss_info = self._compute_loss_safe(predictions, labels)
                
                total_loss += loss_info['total_loss'].item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 计算回归评估指标
        metrics = self._calculate_regression_metrics(all_predictions, all_labels)
        metrics['total_loss'] = total_loss / num_batches
        
        return metrics
    
    def _compute_loss_safe(self, predictions, labels):
        """安全地计算损失，处理混合精度兼容性问题"""
        # 确保float32类型
        if predictions.dtype == torch.float16:
            predictions = predictions.float()
        if labels.dtype == torch.float16:
            labels = labels.float()
        
        # 计算损失
        loss_dict = self.model.compute_loss(predictions, labels)
        
        return loss_dict
    
    def _calculate_regression_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """计算回归评估指标"""
        metrics = {}
        
        # 确保预测值在合理范围内
        predictions = np.clip(predictions, 0.0, 1.0)
        
        # 整体指标
        mae = np.mean(np.abs(predictions - labels))
        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)
        
        metrics['mae'] = mae
        metrics['mse'] = mse
        metrics['rmse'] = rmse
        
        # 计算相关系数
        if np.std(predictions) > 0 and np.std(labels) > 0:
            correlation = np.corrcoef(predictions.flatten(), labels.flatten())[0, 1]
        else:
            correlation = 0.0
        metrics['correlation'] = correlation
        
        # 计算每个任务的指标
        task_metrics = {}
        for i, task_idx in enumerate(self.valid_tasks):
            task_preds = predictions[:, task_idx]
            task_labels = labels[:, task_idx]
            
            # 任务级别的指标
            task_mae = np.mean(np.abs(task_preds - task_labels))
            task_rmse = np.sqrt(np.mean((task_preds - task_labels) ** 2))
            
            # 相关系数
            if np.std(task_preds) > 0 and np.std(task_labels) > 0:
                task_corr = np.corrcoef(task_preds, task_labels)[0, 1]
            else:
                task_corr = 0.0
            
            # 高概率事件的准确性
            high_prob_mask = task_labels > 0.3
            if np.any(high_prob_mask):
                high_prob_mae = np.mean(np.abs(task_preds[high_prob_mask] - task_labels[high_prob_mask]))
            else:
                high_prob_mae = 0.0
            
            task_metrics[f'task_{task_idx}'] = {
                'mae': task_mae,
                'rmse': task_rmse,
                'correlation': task_corr,
                'high_prob_mae': high_prob_mae,
                'mean_pred': np.mean(task_preds),
                'mean_label': np.mean(task_labels),
                'std_pred': np.std(task_preds),
                'std_label': np.std(task_labels)
            }
        
        # 汇总指标
        all_maes = [tm['mae'] for tm in task_metrics.values()]
        all_correlations = [tm['correlation'] for tm in task_metrics.values()]
        
        metrics['mean_task_mae'] = np.mean(all_maes)
        metrics['mean_correlation'] = np.mean(all_correlations)
        
        # 大地震指标
        high_magnitude_indices = [3, 7, 11]  # M6.5+的任务
        high_mag_maes = []
        high_mag_corrs = []
        
        for task_idx in high_magnitude_indices:
            if task_idx in self.valid_tasks:
                task_key = f'task_{task_idx}'
                if task_key in task_metrics:
                    high_mag_maes.append(task_metrics[task_key]['mae'])
                    high_mag_corrs.append(task_metrics[task_key]['correlation'])
        
        metrics['high_magnitude_mae'] = np.mean(high_mag_maes) if high_mag_maes else 0.0
        metrics['high_magnitude_correlation'] = np.mean(high_mag_corrs) if high_mag_corrs else 0.0
        
        # 保存任务级别的指标
        metrics['task_metrics'] = task_metrics
        
        # 打印详细指标（每10个epoch）
        if hasattr(self, 'epoch') and (self.epoch + 1) % 10 == 0:
            print("\n  📊 任务级别详细指标:")
            task_names = [
                "7d_M3-4.5", "7d_M4.5-5.5", "7d_M5.5-6.5", "7d_M6.5+",
                "14d_M3-4.5", "14d_M4.5-5.5", "14d_M5.5-6.5", "14d_M6.5+",
                "30d_M3-4.5", "30d_M4.5-5.5", "30d_M5.5-6.5", "30d_M6.5+"
            ]
            
            for task_idx in self.valid_tasks:
                if f'task_{task_idx}' in task_metrics:
                    tm = task_metrics[f'task_{task_idx}']
                    print(f"    任务{task_idx:2d} ({task_names[task_idx]}): "
                          f"MAE={tm['mae']:.4f}, RMSE={tm['rmse']:.4f}, "
                          f"相关性={tm['correlation']:.3f}")
        
        return metrics
    
    def _calculate_comprehensive_score(self, metrics: Dict[str, float]) -> float:
        """计算综合评分 - 回归版本（越小越好）"""
        # 基于MAE和相关系数的综合评分
        mae_score = metrics.get('mae', 1.0)
        correlation_score = 1.0 - metrics.get('mean_correlation', 0.0)  # 转换为越小越好
        high_mag_score = metrics.get('high_magnitude_mae', 1.0)
        
        # 加权组合
        weights = {
            'mae': 0.4,
            'correlation': 0.3,
            'high_magnitude': 0.3
        }
        
        score = (mae_score * weights['mae'] + 
                 correlation_score * weights['correlation'] + 
                 high_mag_score * weights['high_magnitude'])
        
        return score
    
    def _should_early_stop(self) -> bool:
        """早停判断 - 回归版本"""
        patience = self.config.get('early_stop_patience', 20)
        
        if len(self.val_scores) < patience:
            return False
        
        # 检查最近的分数是否有改善（对于回归，分数越小越好）
        recent_scores = self.val_scores[-patience:]
        best_recent = min(recent_scores)
        
        # 如果最近patience个epoch都没有改善，则早停
        if best_recent >= self.best_score:
            return True
        
        return False
    
    def train(self):
        """完整训练流程"""
        print(f"\n🚀 开始训练 - 总计 {self.config['epochs']} 轮")
        print("="*80)
        
        if torch.cuda.is_available():
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            GPUMemoryMonitor.log_memory_usage("训练开始前", self.device)
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            # 训练
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['total_loss'])
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 验证
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['total_loss'])
            self.val_metrics_history.append(val_metrics)
            
            # 计算综合评分（越小越好）
            comprehensive_score = self._calculate_comprehensive_score(val_metrics)
            self.val_scores.append(comprehensive_score)
            
            # 学习率调度（除了OneCycle）
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['mae'])  # 使用MAE作为监控指标
                else:
                    self.scheduler.step()
            
            # 打印结果
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n📊 Epoch {epoch+1} 结果:")
            print(f"  训练损失: {train_metrics['total_loss']:.4f}")
            print(f"  训练MAE: {train_metrics['mae']:.4f}")
            print(f"  训练RMSE: {train_metrics['rmse']:.4f}")
            print(f"  验证损失: {val_metrics['total_loss']:.4f}")
            print(f"  验证指标:")
            print(f"    MAE: {val_metrics['mae']:.4f}")
            print(f"    RMSE: {val_metrics['rmse']:.4f}")
            print(f"    相关系数: {val_metrics['correlation']:.4f}")
            print(f"    大地震MAE: {val_metrics['high_magnitude_mae']:.4f}")
            print(f"  综合评分: {comprehensive_score:.4f} (越小越好)")
            print(f"  学习率: {current_lr:.6f}")
            print(f"  耗时: {epoch_time:.1f}秒")
            
            # 保存最佳模型（分数越小越好）
            is_best = comprehensive_score < self.best_score
            if is_best:
                self.best_score = comprehensive_score
                self._save_best_model(val_metrics)
                print(f"  🏆 新的最佳模型! 评分: {comprehensive_score:.4f}")
            
            # 定期保存检查点
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self._save_checkpoint(val_metrics)
            
            # 早停检查
            if self._should_early_stop():
                print(f"⏹️  早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成! 总耗时: {total_time/3600:.2f}小时")
        print(f"   最佳评分: {self.best_score:.4f}")
        
        if torch.cuda.is_available():
            GPUMemoryMonitor.log_memory_usage("训练完成", self.device)
        
        self._generate_training_report()
    
    def _save_best_model(self, metrics: Dict[str, float]):
        """保存最佳模型"""
        save_path = os.path.join(self.save_dir, 'best_model.pth')
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            loss=metrics['total_loss'],
            metrics=metrics,
            save_path=save_path,
            is_best=True
        )

        # 保存详细的任务指标
        metrics_path = os.path.join(self.save_dir, 'best_model_metrics.json')
        
        # 转换所有numpy类型为Python原生类型
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            else:
                return obj
        
        metrics_to_save = {
            'epoch': int(self.epoch),
            'comprehensive_score': float(self.best_score),
            'overall_metrics': {k: convert_to_native(v) for k, v in metrics.items() if k != 'task_metrics'},
            'task_metrics': convert_to_native(metrics.get('task_metrics', {}))
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
    
    def _save_checkpoint(self, metrics: Dict[str, float]):
        """保存检查点"""
        save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{self.epoch+1}.pth')
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            loss=metrics['total_loss'],
            metrics=metrics,
            save_path=save_path,
            is_best=False
        )
    
    def _generate_training_report(self):
        """生成训练报告"""
        report_dir = os.path.join(self.save_dir, 'training_report')
        os.makedirs(report_dir, exist_ok=True)
        
        # 绘制损失曲线
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('损失曲线')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.val_scores, label='综合评分')
        plt.xlabel('Epoch')
        plt.ylabel('Score (lower is better)')
        plt.legend()
        plt.title('验证评分')
        plt.grid(True)
        
        # 绘制MAE和相关系数曲线
        plt.subplot(1, 3, 3)
        maes = [m.get('mae', 0) for m in self.val_metrics_history]
        correlations = [m.get('correlation', 0) for m in self.val_metrics_history]
        
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        ax1.plot(maes, 'b-', label='MAE')
        ax2.plot(correlations, 'r-', label='Correlation')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MAE', color='b')
        ax2.set_ylabel('Correlation', color='r')
        
        ax1.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('回归指标')
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'training_curves.png'), dpi=300)
        plt.close()
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_scores': self.val_scores,
            'val_metrics_history': self.val_metrics_history,
            'best_score': self.best_score,
            'config': self.config
        }
        
        history_path = os.path.join(report_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        print(f"📈 训练报告已保存到: {report_dir}")

def create_adaptive_model(data_shape: Tuple[int, ...]) -> EarthquakePredictionModel:
    """根据数据形状创建自适应的模型 - 保持不变"""
    _, time_steps, height, width, channels = data_shape
    
    print(f"📐 根据数据形状创建模型:")
    print(f"  输入通道: {channels}")
    print(f"  时间步: {time_steps}")
    print(f"  网格尺寸: {height} × {width}")

    # 根据7×8网格调整模型容量
    if height == 7 and width == 8:  # 明确处理7×8情况
        spatial_base_channels = 48
        temporal_hidden_channels = 96
        attention_fusion_dim = 384
    # 根据网格大小调整模型容量
    elif height <= 10 and width <= 10:
        # 小网格，使用较大的模型
        spatial_base_channels = 48
        temporal_hidden_channels = 96
        attention_fusion_dim = 384
    elif height <= 20 and width <= 20:
        # 中等网格
        spatial_base_channels = 32
        temporal_hidden_channels = 64
        attention_fusion_dim = 256
    else:
        # 大网格，使用较小的模型
        spatial_base_channels = 24
        temporal_hidden_channels = 48
        attention_fusion_dim = 192
    
    return EarthquakePredictionModel(
        input_channels=channels,
        input_time_steps=time_steps,
        input_height=height,
        input_width=width,
        spatial_base_channels=spatial_base_channels,
        spatial_num_scales=3,
        temporal_hidden_channels=temporal_hidden_channels,
        attention_fusion_dim=attention_fusion_dim,
        shared_feature_dim=64,
        task_hidden_dim=32,
        use_post_processing=True
    )

def create_improved_training_config(mode: str = 'standard') -> Dict[str, Any]:
    """创建改进的训练配置"""
    if mode == 'safe':
        return {
            'batch_size': 2,
            'gradient_accumulation_steps': 16,
            'learning_rate': 2e-4,  # 提高学习率
            'warmup_epochs': 3,
            'optimizer': 'adamw',
            'scheduler': 'cosine',  # 改为cosine
            'weight_decay': 1e-4,
            'grad_clip': 1.0,
            'use_amp': False,
            'epochs': 80,
            'early_stop_patience': 20,
            'scheduler_patience': 8,
            'num_workers': 0,
            'pin_memory': True if torch.cuda.is_available() else False,
            'empty_cache_freq': 3,
            'save_interval': 5,
            'save_dir': 'earthquake_model_improved_safe',
            'loss_type': 'bce',  # 使用BCE损失
            'consistency_weight': 0.05,
            'uncertainty_weighting': True,
            'ignore_tasks': [1],  # 忽略任务1
            'description': '改进的安全模式配置'
        }
    elif mode == 'standard':
        return {
            'batch_size': 4,
            'gradient_accumulation_steps': 8,
            'learning_rate': 5e-4,  # 提高学习率
            'warmup_epochs': 3,
            'optimizer': 'adamw',
            'scheduler': 'onecycle',  # 使用OneCycle
            'weight_decay': 1e-4,
            'grad_clip': 1.0,
            'use_amp': False,
            'epochs': 100,
            'early_stop_patience': 25,
            'scheduler_patience': 10,
            'num_workers': 0,
            'pin_memory': True if torch.cuda.is_available() else False,
            'empty_cache_freq': 5,
            'save_interval': 5,
            'save_dir': 'earthquake_model_improved_standard',
            'loss_type': 'bce',  # 使用BCE损失
            'consistency_weight': 0.05,
            'uncertainty_weighting': True,
            'ignore_tasks': [1],  # 忽略任务1
            'description': '改进的标准模式配置'
        }
    else:  # performance
        return {
            'batch_size': 6,
            'gradient_accumulation_steps': 6,
            'learning_rate': 1e-3,  # 更高的学习率
            'warmup_epochs': 2,
            'optimizer': 'adamw',
            'scheduler': 'onecycle',  # 使用OneCycle
            'weight_decay': 1e-4,
            'grad_clip': 1.0,
            'use_amp': True,  # 启用混合精度
            'epochs': 100,
            'early_stop_patience': 30,
            'scheduler_patience': 10,
            'num_workers': 0,
            'pin_memory': True if torch.cuda.is_available() else False,
            'empty_cache_freq': 10,
            'save_interval': 5,
            'save_dir': 'earthquake_model_improved_performance',
            'loss_type': 'bce',  # 使用BCE损失
            'consistency_weight': 0.05,
            'uncertainty_weighting': True,
            'ignore_tasks': [1],  # 忽略任务1
            'description': '改进的高性能模式配置'
        }

def main():
    """主训练函数"""
    # GPU优化设置
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("🌍 地震预测模型训练系统 - 改进版")
    print("="*60)
    
    # 显示系统信息
    print("\n📊 系统信息:")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU设备: {torch.cuda.get_device_name()}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
    
    # 数据路径
    data_dir = "../data/processed_grid"
    train_features_path = os.path.join(data_dir, "train_features.npy")
    train_stat_features_path = os.path.join(data_dir, "train_stat_features.npy")
    train_labels_path = os.path.join(data_dir, "train_labels.npy")
    train_timestamps_path = os.path.join(data_dir, "train_timestamps.npy")
    train_weighted_labels_path = os.path.join(data_dir, "train_weighted_labels.npy")
    
    val_features_path = os.path.join(data_dir, "val_features.npy")
    val_stat_features_path = os.path.join(data_dir, "val_stat_features.npy")
    val_labels_path = os.path.join(data_dir, "val_labels.npy")
    val_timestamps_path = os.path.join(data_dir, "val_timestamps.npy")
    val_weighted_labels_path = os.path.join(data_dir, "val_weighted_labels.npy")
    
    # 检查文件
    required_files = [
        train_features_path, train_stat_features_path, train_labels_path, train_timestamps_path,
        val_features_path, val_stat_features_path, val_labels_path, val_timestamps_path
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ 缺少以下数据文件:")
        for f in missing_files:
            print(f"  - {f}")
        return
    
    # 检查数据形状
    print("\n📏 检查数据形状...")
    sample_features = np.load(train_features_path, mmap_mode='r')
    data_shape = sample_features.shape
    print(f"  数据形状: {data_shape}")
    
    # 创建数据集
    print("\n📂 加载数据集...")
    import platform
    use_mmap = platform.system() == 'Windows'
    
    train_dataset = EarthquakeDataset(
        features_path=train_features_path,
        stat_features_path=train_stat_features_path,
        labels_path=train_labels_path,
        timestamps_path=train_timestamps_path,
        sample_weights_path=val_weighted_labels_path,
        augment=True,
        balance_sampling=True,
        use_mmap=use_mmap
    )
    
    val_dataset = EarthquakeDataset(
        features_path=val_features_path,
        stat_features_path=val_stat_features_path,
        labels_path=val_labels_path,
        timestamps_path=val_timestamps_path,
        sample_weights_path=train_weighted_labels_path,
        augment=False,
        balance_sampling=False,
        use_mmap=use_mmap
    )
    
    # 创建自适应模型
    print("\n🏗️ 创建地震预测模型...")
    model = create_adaptive_model(data_shape)
    
    # 选择训练模式
    if torch.cuda.is_available():
        print("\n请选择训练模式:")
        print("1. 安全模式 (批次2, 梯度累积16, 学习率2e-4)")
        print("2. 标准模式 (批次4, 梯度累积8, 学习率5e-4)")
        print("3. 高性能模式 (批次6, 梯度累积6, 学习率1e-3)")
        
        mode_input = input("请输入选择 (1/2/3, 默认为2): ").strip() or "2"
        
        mode_map = {'1': 'safe', '2': 'standard', '3': 'performance'}
        mode = mode_map.get(mode_input, 'standard')
    else:
        mode = 'safe'
    
    config = create_improved_training_config(mode)
    
    print(f"\n⚙️ 训练配置 ({mode}模式):")
    for key, value in config.items():
        if key != 'description':
            print(f"  {key}: {value}")
    
    # 显示模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型参数量:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 创建训练器
    trainer = ImprovedEarthquakeTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # 开始训练
    try:
        trainer.train()
        print("\n🎉 训练成功完成!")
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        trainer._save_checkpoint({'total_loss': float('inf')})
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ GPU内存不足！")
            print("建议: 使用安全模式重新运行")
        else:
            print(f"\n❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()