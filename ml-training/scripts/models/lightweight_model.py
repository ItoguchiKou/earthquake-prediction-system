"""
轻量级地震预测模型
专门针对不规则网格设计 - 回归版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional

class IrregularGridMaskAttention(nn.Module):
    """不规则网格掩码注意力模块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # 创建有效网格掩码（硬编码28个有效网格）
        self.register_buffer('valid_mask', self._create_valid_mask())
    
    def _create_valid_mask(self) -> torch.Tensor:
        """创建有效网格掩码"""
        mask = torch.zeros(1, 1, 10, 8)
        # 根据28个有效网格设置掩码
        valid_positions = [
            (0, 4), # 冲绳
            (1, 2), (1, 3), # 九州南部
            (2, 2), (2, 3), # 九州中部
            (3, 2), (3, 3), (3, 4), (3, 5), # 九州北部・中国西部
            (4, 2), (4, 3), (4, 4), (4, 5), # 中国・近畿
            (5, 3), (5, 4), (5, 5), (5, 6), # 中部
            (6, 4), (6, 5), (6, 6), # 北陆・东北南部
            (7, 4), (7, 5), (7, 6), # 东北中部
            (8, 5), (8, 6), # 东北北部
            (9, 5), (9, 6), (9, 7), # 北海道
        ]
        for lat, lon in valid_positions:
            mask[0, 0, lat, lon] = 1.0
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算注意力权重
        attention_weights = self.attention(x)
        # 应用有效网格掩码
        attention_weights = attention_weights * self.valid_mask
        # 应用注意力
        return x * attention_weights + x

class SimplifiedSpatialEncoder(nn.Module):
    """简化的空间编码器 - 适配不规则网格"""
    
    def __init__(self, 
                 input_channels: int = 8,
                 hidden_channels: int = 16,
                 output_channels: int = 32):
        super().__init__()
        
        # 简单的卷积层
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        # 不规则网格注意力
        self.grid_attention = IrregularGridMaskAttention(hidden_channels)
        
        self.conv2 = nn.Conv2d(hidden_channels, output_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # x: [batch, time, height, width, channels]
        b, t, h, w, c = x.shape
        
        # 重塑为2D卷积格式
        x = x.view(b * t, c, h, w)
        
        # 卷积处理
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 应用网格注意力
        x = self.grid_attention(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 全局池化
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        # 重塑回时间序列
        x = x.view(b, t, -1)
        
        return x

class SimplifiedTemporalEncoder(nn.Module):
    """简化的时间编码器"""
    
    def __init__(self,
                 input_dim: int = 32,
                 hidden_dim: int = 64,
                 num_layers: int = 1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 输出维度是hidden_dim * 2（双向）
        self.output_dim = hidden_dim * 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # x: [batch, time, features]
        lstm_out, _ = self.lstm(x)
        
        # 使用最后时间步的输出
        return lstm_out[:, -1, :]

class TaskSpecificHead(nn.Module):
    """任务特定输出头 - 回归版本"""
    
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 32,
                 num_tasks: int = 12,
                 valid_tasks: List[int] = None,
                 output_activation: str = 'none'):  # 'none', 'sigmoid', 'tanh'
        super().__init__()
        
        self.num_tasks = num_tasks
        self.valid_tasks = valid_tasks or list(range(num_tasks))
        self.output_activation = output_activation
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 每个有效任务一个输出头
        self.task_heads = nn.ModuleDict()
        for task_idx in self.valid_tasks:
            self.task_heads[str(task_idx)] = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 共享特征
        shared_features = self.shared(x)
        
        # 任务输出
        outputs = []
        for task_idx in range(self.num_tasks):
            if task_idx in self.valid_tasks:
                task_output = self.task_heads[str(task_idx)](shared_features)
                
                # 应用激活函数（如果需要）
                if self.output_activation == 'sigmoid':
                    task_output = torch.sigmoid(task_output)
                elif self.output_activation == 'tanh':
                    task_output = (torch.tanh(task_output) + 1) / 2  # 映射到[0,1]
                # 默认不使用激活函数，让模型自由学习
                
            else:
                # 无效任务输出零
                task_output = torch.zeros(x.size(0), 1, device=x.device)
            outputs.append(task_output)
        
        # 拼接所有任务输出
        return torch.cat(outputs, dim=1)

class LightweightEarthquakeModel(nn.Module):
    """轻量级地震预测模型 - 不规则网格版本"""
    
    def __init__(self,
                 input_channels: int = 8,
                 input_time_steps: int = 90,
                 input_height: int = 10,  # 改为10
                 input_width: int = 8,    # 改为8
                 spatial_hidden: int = 16,
                 spatial_output: int = 32,
                 temporal_hidden: int = 64,
                 task_hidden: int = 32,
                 num_tasks: int = 12,
                 valid_tasks: List[int] = None,
                 use_stat_features: bool = True,
                 output_activation: str = 'none'):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_time_steps = input_time_steps
        self.input_height = input_height
        self.input_width = input_width
        self.use_stat_features = use_stat_features
        
        # 空间编码器（带不规则网格注意力）
        self.spatial_encoder = SimplifiedSpatialEncoder(
            input_channels=input_channels,
            hidden_channels=spatial_hidden,
            output_channels=spatial_output
        )
        
        # 时间编码器
        self.temporal_encoder = SimplifiedTemporalEncoder(
            input_dim=spatial_output,
            hidden_dim=temporal_hidden,
            num_layers=1
        )
        
        # 统计特征处理（如果使用）
        if use_stat_features:
            # 统计特征维度：8通道 × 6统计量 = 48
            stat_feature_dim = input_channels * 6
            self.stat_processor = nn.Sequential(
                nn.Linear(input_height * input_width * stat_feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64)
            )
            final_feature_dim = temporal_hidden * 2 + 64  # LSTM是双向的
        else:
            self.stat_processor = None
            final_feature_dim = temporal_hidden * 2
        
        # 任务输出头
        self.task_head = TaskSpecificHead(
            input_dim=final_feature_dim,
            hidden_dim=task_hidden,
            num_tasks=num_tasks,
            valid_tasks=valid_tasks or list(range(num_tasks)),
            output_activation=output_activation
        )
        
        # 回归任务的权重初始化
        self._initialize_weights()

        # 总参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"轻量级不规则网格模型初始化完成:")
        print(f"  输入形状: [{input_time_steps}, {input_height}, {input_width}, {input_channels}]")
        print(f"  总参数量: {total_params:,}")
        print(f"  有效任务: {self.task_head.valid_tasks}")
        print(f"  输出激活: {output_activation}")
    
    def _initialize_weights(self):
        """回归任务的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.out_features == 1:  # 任务输出层
                    # 回归任务：使用更小的初始化
                    nn.init.normal_(m.weight, 0, 0.01)
                    # 根据数据的平均概率初始化偏置
                    nn.init.constant_(m.bias, 0.2)  # 接近数据的平均值
                else:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
                            
    def forward(self, x: torch.Tensor, stat_features: Optional[torch.Tensor] = None, temperature: float = 1.0) -> torch.Tensor:
        """前向传播"""
        # 空间编码
        spatial_features = self.spatial_encoder(x)
        
        # 时间编码
        temporal_features = self.temporal_encoder(spatial_features)
        
        # 处理统计特征
        if self.use_stat_features and stat_features is not None:
            # 展平统计特征
            batch_size = stat_features.size(0)
            stat_features_flat = stat_features.view(batch_size, -1)
            stat_features_processed = self.stat_processor(stat_features_flat)
            
            # 融合特征
            combined_features = torch.cat([temporal_features, stat_features_processed], dim=1)
        else:
            combined_features = temporal_features
        
        # 任务预测
        predictions = self.task_head(combined_features)
        
        # 温度缩放（用于推理时的不确定性调整）
        if temperature != 1.0:
            predictions = predictions * temperature
        
        # 确保输出在合理范围内（0-1）
        predictions = torch.clamp(predictions, 0.0, 1.0)
        
        return predictions
    
    def forward_with_stat_features(self, x: torch.Tensor, stat_features: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """带统计特征的前向传播"""
        return self.forward(x, stat_features, temperature)

def create_lightweight_model(data_shape: Tuple[int, ...],
                           model_type: str = 'standard',
                           valid_tasks: List[int] = None) -> nn.Module:
    """
    创建轻量级模型
    
    Args:
        data_shape: 数据形状
        model_type: 模型类型 ('standard', 'minimal', 'regional')
        valid_tasks: 有效任务列表
        
    Returns:
        模型实例
    """
    _, time_steps, height, width, channels = data_shape
    
    if valid_tasks is None:
        valid_tasks = list(range(12))  # 不再排除任何任务
    
    if model_type == 'minimal':
        # 最小化模型
        return LightweightEarthquakeModel(
            input_channels=channels,
            input_time_steps=time_steps,
            input_height=height,
            input_width=width,
            spatial_hidden=8,
            spatial_output=16,
            temporal_hidden=32,
            task_hidden=16,
            num_tasks=12,
            valid_tasks=valid_tasks,
            use_stat_features=False,
            output_activation='none'  # 回归任务不使用激活
        )
    elif model_type == 'standard':
        # 标准轻量级模型
        return LightweightEarthquakeModel(
            input_channels=channels,
            input_time_steps=time_steps,
            input_height=height,
            input_width=width,
            spatial_hidden=16,
            spatial_output=32,
            temporal_hidden=64,
            task_hidden=32,
            num_tasks=12,
            valid_tasks=valid_tasks,
            use_stat_features=True,
            output_activation='none'  # 回归任务不使用激活
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

# 测试代码
if __name__ == "__main__":
    # 测试轻量级模型
    print("测试轻量级不规则网格模型:")
    
    # 模拟数据 - 注意维度改变
    batch_size = 4
    time_steps = 90
    height, width = 10, 8  # 改为10×8
    channels = 8
    
    x = torch.randn(batch_size, time_steps, height, width, channels)
    stat_features = torch.randn(batch_size, height, width, 48)  # 8通道×6统计量
    
    # 创建模型
    model = create_lightweight_model(
        data_shape=(batch_size, time_steps, height, width, channels),
        model_type='standard'
    )
    
    # 前向传播
    with torch.no_grad():
        predictions = model.forward_with_stat_features(x, stat_features)
        print(f"预测输出形状: {predictions.shape}")
        print(f"预测值范围: [{predictions.min():.3f}, {predictions.max():.3f}]")
        print(f"预测均值: {predictions.mean():.3f}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")