"""
损失函数定义模块 - 回归版本
针对地震概率预测的回归任务优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

class HuberLoss(nn.Module):
    """Huber Loss - 对异常值更鲁棒的回归损失"""
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        diff = torch.abs(predictions - targets)
        
        # Huber loss: 小误差用L2，大误差用L1
        mask = diff < self.delta
        loss = torch.where(
            mask,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class WeightedMSELoss(nn.Module):
    """加权MSE损失 - 对不同概率范围给予不同权重"""
    
    def __init__(self, 
                 probability_weights: Optional[torch.Tensor] = None,
                 task_weights: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.probability_weights = probability_weights
        self.task_weights = task_weights
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 基本MSE
        mse = (predictions - targets) ** 2
        
        # 根据目标概率值加权（高概率事件更重要）
        if self.probability_weights is not None:
            # 根据目标值计算权重
            weights = 1.0 + targets * self.probability_weights
            mse = mse * weights
        
        # 任务级别加权
        if self.task_weights is not None:
            mse = mse * self.task_weights.unsqueeze(0)
        
        if self.reduction == 'mean':
            return mse.mean()
        elif self.reduction == 'sum':
            return mse.sum()
        else:
            return mse

class ProbabilityRegressionLoss(nn.Module):
    """专门为概率回归设计的损失函数"""
    
    def __init__(self,
                 base_loss: str = 'huber',  # 'mse', 'huber', 'mae'
                 huber_delta: float = 0.1,
                 probability_threshold: float = 0.3,  # 重要事件阈值
                 high_prob_weight: float = 5.0,  # 高概率事件权重
                 consistency_weight: float = 0.1):
        super().__init__()
        
        self.base_loss = base_loss
        self.huber_delta = huber_delta
        self.probability_threshold = probability_threshold
        self.high_prob_weight = high_prob_weight
        self.consistency_weight = consistency_weight
        
        # 基础损失函数
        if base_loss == 'huber':
            self.loss_fn = HuberLoss(delta=huber_delta, reduction='none')
        elif base_loss == 'mae':
            self.loss_fn = nn.L1Loss(reduction='none')
        else:  # mse
            self.loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 基础损失
        base_loss = self.loss_fn(predictions, targets)
        
        # 对高概率事件加权
        high_prob_mask = targets > self.probability_threshold
        weights = torch.ones_like(targets)
        weights[high_prob_mask] = self.high_prob_weight
        
        weighted_loss = base_loss * weights
        
        # 时间一致性损失（可选）
        if self.consistency_weight > 0:
            consistency_loss = self._compute_consistency_loss(predictions)
            total_loss = weighted_loss.mean() + self.consistency_weight * consistency_loss
        else:
            total_loss = weighted_loss.mean()
        
        return total_loss
    
    def _compute_consistency_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """计算时间一致性损失"""
        batch_size = predictions.size(0)
        pred_reshaped = predictions.view(batch_size, 3, 4)  # 3个时间窗口，4个震级
        
        consistency_loss = 0.0
        
        # 时间一致性：后续时间窗口的概率应该 >= 前面的
        for t in range(2):
            # 使用软约束而不是硬约束
            diff = F.relu(pred_reshaped[:, t, :] - pred_reshaped[:, t+1, :])
            consistency_loss = consistency_loss + diff.mean()
        
        # 震级一致性：大震级的概率应该 <= 小震级
        for t in range(3):
            for m in range(3):
                diff = F.relu(pred_reshaped[:, t, m+1] - pred_reshaped[:, t, m])
                consistency_loss = consistency_loss + diff.mean() * 0.5
        
        return consistency_loss

class EarthquakeRegressionLoss(nn.Module):
    """地震预测回归综合损失函数"""
    
    def __init__(self,
                 loss_type: str = 'huber',
                 task_weights: Optional[List[float]] = None,
                 consistency_weight: float = 0.1,
                 ignore_tasks: Optional[List[int]] = None):
        """
        初始化回归损失函数
        
        Args:
            loss_type: 损失类型 ('mse', 'huber', 'mae', 'weighted_mse')
            task_weights: 任务权重
            consistency_weight: 一致性损失权重
            ignore_tasks: 要忽略的任务列表
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.consistency_weight = consistency_weight
        self.ignore_tasks = ignore_tasks or []
        
        # 基于任务统计设置默认权重
        if task_weights is None:
            # 基于平均概率值的反比例权重
            task_mean_probs = torch.tensor([
                0.0796, 0.2516, 0.1322, 0.0274,  # 7天
                0.0996, 0.3163, 0.2228, 0.0529,  # 14天
                0.1275, 0.4163, 0.3453, 0.1047   # 30天
            ])
            # 反比例权重，并进行归一化
            self.task_weights = 1.0 / (task_mean_probs + 0.01)
            self.task_weights = self.task_weights / self.task_weights.mean()
        else:
            self.task_weights = torch.tensor(task_weights)
        
        # 将忽略任务的权重设为0
        for idx in self.ignore_tasks:
            self.task_weights[idx] = 0.0
        
        # 创建主损失函数
        if loss_type == 'huber':
            self.main_loss = ProbabilityRegressionLoss(
                base_loss='huber',
                huber_delta=0.1,
                consistency_weight=0
            )
        elif loss_type == 'weighted_mse':
            self.main_loss = WeightedMSELoss(
                probability_weights=torch.tensor(5.0),
                task_weights=self.task_weights
            )
        elif loss_type == 'mae':
            self.main_loss = nn.L1Loss(reduction='none')
        else:  # mse
            self.main_loss = nn.MSELoss(reduction='none')
        
        # 一致性损失
        if consistency_weight > 0:
            self.consistency_loss = ConsistencyLoss(weight=consistency_weight)
        
        print(f"📊 地震回归损失初始化:")
        print(f"  损失类型: {loss_type}")
        print(f"  一致性权重: {consistency_weight}")
        print(f"  忽略任务: {self.ignore_tasks}")
        print(f"  任务权重范围: [{self.task_weights.min():.2f}, {self.task_weights.max():.2f}]")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算综合损失"""
        device = predictions.device
        batch_size = predictions.size(0)
        
        # 计算主损失
        if isinstance(self.main_loss, (nn.MSELoss, nn.L1Loss)):
            task_losses = self.main_loss(predictions, targets)
            # 应用任务权重
            task_losses = task_losses * self.task_weights.to(device)
            main_loss = task_losses.mean()
        else:
            main_loss = self.main_loss(predictions, targets)
            task_losses = torch.zeros(12, device=device)  # 简化版，不分解到任务
        
        # 一致性损失
        if self.consistency_weight > 0 and hasattr(self, 'consistency_loss'):
            consistency_loss = self.consistency_loss(predictions)
            total_loss = main_loss + consistency_loss
        else:
            consistency_loss = torch.tensor(0.0, device=device)
            total_loss = main_loss
        
        # 计算额外的回归指标
        with torch.no_grad():
            mae = torch.abs(predictions - targets).mean()
            rmse = torch.sqrt(((predictions - targets) ** 2).mean())
            
            # 计算相关系数（简化版）
            pred_mean = predictions.mean()
            target_mean = targets.mean()
            pred_std = predictions.std()
            target_std = targets.std()
            
            if pred_std > 0 and target_std > 0:
                correlation = ((predictions - pred_mean) * (targets - target_mean)).mean() / (pred_std * target_std)
            else:
                correlation = torch.tensor(0.0, device=device)
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'consistency_loss': consistency_loss,
            'task_losses': task_losses,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation
        }

class ConsistencyLoss(nn.Module):
    """时间和震级一致性损失"""
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """计算一致性损失"""
        batch_size = predictions.size(0)
        pred_reshaped = predictions.view(batch_size, 3, 4)
        
        consistency_loss = 0.0
        
        # 时间一致性（软约束）
        for t in range(2):
            # 14天/30天的概率应该趋向于大于等于7天/14天
            diff = F.relu(pred_reshaped[:, t, :] - pred_reshaped[:, t+1, :] - 0.05)  # 允许小的反转
            consistency_loss = consistency_loss + diff.mean()
        
        # 震级一致性（软约束）
        for t in range(3):
            for m in range(3):
                # 大震级概率应该小于等于小震级
                diff = F.relu(pred_reshaped[:, t, m+1] - pred_reshaped[:, t, m] - 0.02)  # 允许小的反转
                consistency_loss = consistency_loss + diff.mean() * 0.5
        
        return consistency_loss * self.weight

# 向后兼容的别名
EarthquakeLoss = EarthquakeRegressionLoss
BalancedBCELoss = WeightedMSELoss  # 简单映射，实际已改为MSE
ImprovedMultiTaskLoss = EarthquakeRegressionLoss
ImprovedFocalLoss = HuberLoss  # 简单映射