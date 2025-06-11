"""
多任务输出头模块
实现12个地震预测任务的共享特征提取和任务特定输出头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class SharedFeatureExtractor(nn.Module):
    """
    共享特征提取器 - 为所有预测任务提供共同的特征表示
    """

    def __init__(self,
                 input_channels: int = 512,
                 hidden_channels: int = 256,
                 output_channels: int = 128,
                 dropout_rate: float = 0.2):
        """
        初始化共享特征提取器

        Args:
            input_channels: 输入特征通道数
            hidden_channels: 隐藏层通道数
            output_channels: 输出特征通道数
            dropout_rate: Dropout比例
        """
        super(SharedFeatureExtractor, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            # 第一层：降维并提取抽象特征
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            # 第二层：进一步特征抽象
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            # 第三层：输出特征
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        # 全局特征聚合
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        # 特征压缩到任务特定维度
        self.feature_compression = nn.Sequential(
            nn.Linear(output_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, output_channels)
        )

        print(f":wrench: 共享特征提取器初始化:")
        print(f"  输入通道: {input_channels}")
        print(f"  隐藏通道: {hidden_channels}")
        print(f"  输出通道: {output_channels}")
        print(f"  Dropout率: {dropout_rate}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征 [batch, time_steps, channels, height, width]

        Returns:
            (空间特征, 全局特征)
        """
        batch_size, time_steps, channels, height, width = x.size()

        # 逐帧处理
        spatial_features = []
        global_features = []

        for t in range(time_steps):
            # 提取空间特征
            spatial_feat = self.feature_extractor(x[:, t, :, :, :])
            spatial_features.append(spatial_feat)

            # 提取全局特征
            global_feat = self.global_pooling(spatial_feat).squeeze(-1).squeeze(-1)
            global_feat = self.feature_compression(global_feat)
            global_features.append(global_feat)

        # 堆叠时间序列
        spatial_features = torch.stack(spatial_features, dim=1)
        global_features = torch.stack(global_features, dim=1)

        return spatial_features, global_features

class TaskSpecificHead(nn.Module):
    """
    任务特定输出头 - 为单个预测任务设计的专用网络
    """

    def __init__(self,
                 feature_dim: int = 128,
                 hidden_dim: int = 64,
                 task_name: str = "task",
                 use_spatial_info: bool = True):
        """
        初始化任务特定头

        Args:
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
            task_name: 任务名称
            use_spatial_info: 是否使用空间信息
        """
        super(TaskSpecificHead, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.task_name = task_name
        self.use_spatial_info = use_spatial_info

        if use_spatial_info:
            # 空间信息处理
            self.spatial_processor = nn.Sequential(
                nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            input_dim = feature_dim + hidden_dim  # 全局特征 + 空间特征
        else:
            input_dim = feature_dim  # 仅全局特征

        # 时间信息处理
        self.temporal_processor = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )

        # 最终预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出概率
        )

        print(f":dart: 任务头 '{task_name}' 初始化: {feature_dim}维 → {hidden_dim}维 → 1维")

    def forward(self,
                global_features: torch.Tensor,
                spatial_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            global_features: 全局特征 [batch, time_steps, feature_dim]
            spatial_features: 空间特征 [batch, time_steps, feature_dim, height, width]

        Returns:
            任务预测 [batch, 1]
        """
        batch_size, time_steps, feature_dim = global_features.size()

        if self.use_spatial_info and spatial_features is not None:
            # 处理空间特征
            spatial_processed = []
            for t in range(time_steps):
                spatial_feat = self.spatial_processor(spatial_features[:, t, :, :, :])
                spatial_processed.append(spatial_feat)
            spatial_processed = torch.stack(spatial_processed, dim=1)

            # 融合全局和空间特征
            combined_features = torch.cat([global_features, spatial_processed], dim=-1)
        else:
            combined_features = global_features

        # 时间序列处理
        lstm_output, _ = self.temporal_processor(combined_features)

        # 使用最后时间步的输出进行预测
        final_output = lstm_output[:, -1, :]

        # 生成预测
        prediction = self.predictor(final_output)

        return prediction

class MultiTaskOutputLayer(nn.Module):
    """
    多任务输出层 - 管理12个地震预测任务
    """

    def __init__(self,
                 input_channels: int = 512,
                 shared_feature_dim: int = 128,
                 task_hidden_dim: int = 64,
                 prediction_windows: List[int] = [7, 14, 30],
                 magnitude_ranges: List[Tuple[float, float]] = [(3.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 10.0)]):
        """
        初始化多任务输出层

        Args:
            input_channels: 输入特征通道数
            shared_feature_dim: 共享特征维度
            task_hidden_dim: 任务隐藏层维度
            prediction_windows: 预测时间窗口(天)
            magnitude_ranges: 震级范围
        """
        super(MultiTaskOutputLayer, self).__init__()

        self.input_channels = input_channels
        self.shared_feature_dim = shared_feature_dim
        self.task_hidden_dim = task_hidden_dim
        self.prediction_windows = prediction_windows
        self.magnitude_ranges = magnitude_ranges

        # 计算任务数量
        self.num_tasks = len(prediction_windows) * len(magnitude_ranges)

        # 创建任务映射
        self.task_map = {}
        self.task_names = []
        task_idx = 0

        for time_window in prediction_windows:
            for mag_min, mag_max in magnitude_ranges:
                task_name = f"T{time_window}d_M{mag_min}-{mag_max}"
                self.task_map[task_idx] = {
                    'name': task_name,
                    'time_window': time_window,
                    'magnitude_range': (mag_min, mag_max)
                }
                self.task_names.append(task_name)
                task_idx += 1

        # 共享特征提取器
        self.shared_extractor = SharedFeatureExtractor(
            input_channels=input_channels,
            hidden_channels=256,
            output_channels=shared_feature_dim,
            dropout_rate=0.2
        )

        # 任务特定输出头
        self.task_heads = nn.ModuleDict()

        for task_idx, task_info in self.task_map.items():
            task_name = task_info['name']

            # 根据任务类型决定是否使用空间信息
            # 短期预测更依赖空间信息，长期预测更依赖全局趋势
            use_spatial = task_info['time_window'] <= 14

            self.task_heads[str(task_idx)] = TaskSpecificHead(
                feature_dim=shared_feature_dim,
                hidden_dim=task_hidden_dim,
                task_name=task_name,
                use_spatial_info=use_spatial
            )

        print(f":dart: 多任务输出层初始化:")
        print(f"  输入通道: {input_channels}")
        print(f"  共享特征维度: {shared_feature_dim}")
        print(f"  任务数量: {self.num_tasks}")
        print(f"  预测窗口: {prediction_windows} 天")
        print(f"  震级范围: {magnitude_ranges}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征 [batch, time_steps, channels, height, width]

        Returns:
            任务预测字典
        """
        # 提取共享特征
        spatial_features, global_features = self.shared_extractor(x)

        # 各任务预测
        predictions = {}
        task_outputs = []

        for task_idx in range(self.num_tasks):
            task_head = self.task_heads[str(task_idx)]
            task_name = self.task_map[task_idx]['name']

            # 根据任务头配置决定输入
            if task_head.use_spatial_info:
                task_pred = task_head(global_features, spatial_features)
            else:
                task_pred = task_head(global_features)

            predictions[task_name] = task_pred
            task_outputs.append(task_pred)

        # 拼接所有任务输出
        all_predictions = torch.cat(task_outputs, dim=1)  # [batch, num_tasks]
        predictions['all_tasks'] = all_predictions

        return predictions

    def get_task_info(self) -> Dict:
        """获取任务信息"""
        return {
            'num_tasks': self.num_tasks,
            'task_map': self.task_map,
            'task_names': self.task_names,
            'prediction_windows': self.prediction_windows,
            'magnitude_ranges': self.magnitude_ranges
        }

    def get_predictions_by_time_window(self, predictions: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        按时间窗口分组预测结果

        Args:
            predictions: 所有任务预测 [batch, num_tasks]

        Returns:
            按时间窗口分组的预测
        """
        grouped_predictions = {}

        for time_idx, time_window in enumerate(self.prediction_windows):
            start_idx = time_idx * len(self.magnitude_ranges)
            end_idx = start_idx + len(self.magnitude_ranges)

            grouped_predictions[time_window] = predictions[:, start_idx:end_idx]

        return grouped_predictions

    def get_predictions_by_magnitude(self, predictions: torch.Tensor) -> Dict[Tuple[float, float], torch.Tensor]:
        """
        按震级范围分组预测结果

        Args:
            predictions: 所有任务预测 [batch, num_tasks]

        Returns:
            按震级范围分组的预测
        """
        grouped_predictions = {}

        for mag_idx, mag_range in enumerate(self.magnitude_ranges):
            # 获取该震级范围在所有时间窗口的预测
            indices = [mag_idx + time_idx * len(self.magnitude_ranges)
                      for time_idx in range(len(self.prediction_windows))]

            grouped_predictions[mag_range] = predictions[:, indices]

        return grouped_predictions

class PredictionPostProcessor(nn.Module):
    """
    预测后处理器 - 对多任务输出进行后处理和一致性约束
    """

    def __init__(self, num_tasks: int = 12):
        """
        初始化预测后处理器

        Args:
            num_tasks: 任务数量
        """
        super(PredictionPostProcessor, self).__init__()

        self.num_tasks = num_tasks

        # 一致性约束权重
        self.consistency_weights = nn.Parameter(torch.ones(num_tasks))

        print(f":wrench: 预测后处理器初始化: {num_tasks} 个任务")

    def apply_consistency_constraints(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        应用时间和震级一致性约束
        
        Args:
            predictions: 原始预测 [batch, 12]
            
        Returns:
            约束后的预测
        """
        batch_size = predictions.size(0)
        
        # 重塑为 [batch_size, 3时间窗口, 4震级类别]
        pred_reshaped = predictions.view(batch_size, 3, 4)
        
        # 创建一个新的张量来存储约束后的结果，而不是修改原张量
        constrained = torch.zeros_like(pred_reshaped)
        
        # 首先复制原始值
        constrained = pred_reshaped.clone()
        
        # 时间一致性约束：使用torch操作而不是循环赋值
        # 14天预测 >= 7天预测
        constrained[:, 1, :] = torch.max(constrained[:, 1, :], constrained[:, 0, :])
        
        # 30天预测 >= 14天预测（注意这里要使用更新后的14天预测）
        constrained = torch.cat([
            constrained[:, 0:1, :],  # 7天预测保持不变
            constrained[:, 1:2, :],  # 14天预测已更新
            torch.max(constrained[:, 2:3, :], constrained[:, 1:2, :])  # 30天预测
        ], dim=1)
        
        # 震级一致性约束：创建新张量
        final_constrained = torch.zeros_like(constrained)
        
        # 对每个时间窗口应用震级约束
        for time_idx in range(3):
            time_slice = constrained[:, time_idx, :]  # [batch, 4]
            
            # M3-5保持不变
            mag_constrained = torch.zeros_like(time_slice)
            mag_constrained[:, 0] = time_slice[:, 0]
            
            # 其他震级不能超过前一个震级
            for mag_idx in range(1, 4):
                mag_constrained[:, mag_idx] = torch.min(
                    time_slice[:, mag_idx], 
                    mag_constrained[:, mag_idx-1]
                )
            
            final_constrained[:, time_idx, :] = mag_constrained
        
        # 重塑回原始形状
        constrained_flat = final_constrained.view(batch_size, self.num_tasks)
        
        # 应用一致性权重
        weighted_predictions = constrained_flat * torch.sigmoid(self.consistency_weights)
        
        return weighted_predictions

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 完整后处理流程
        """
        # 保存梯度信息
        predictions_grad = predictions.requires_grad
        
        # 分离张量进行约束计算
        with torch.no_grad():
            constrained_values = self.apply_consistency_constraints(predictions.detach())
        
        # 创建一个新的张量，保持梯度流
        if predictions_grad:
            # 使用一个可微的方式应用约束
            # 这里使用软约束而不是硬约束
            alpha = 0.8  # 约束强度
            final_predictions = alpha * constrained_values + (1 - alpha) * predictions
            final_predictions = torch.clamp(final_predictions, 0.0, 1.0)
        else:
            final_predictions = torch.clamp(constrained_values, 0.0, 1.0)
        
        return final_predictions

def test_multi_task_heads():
    """测试多任务输出头模块"""
    print(":test_tube: 测试多任务输出头模块...")

    # 模拟输入数据
    batch_size = 2
    time_steps = 10  # 减少时间步以加快测试
    height, width = 22, 24
    input_channels = 512

    input_features = torch.randn(batch_size, time_steps, input_channels, height, width)
    print(f":bar_chart: 输入特征形状: {input_features.shape}")

    # 1. 测试共享特征提取器
    print("\n1. 测试共享特征提取器:")
    shared_extractor = SharedFeatureExtractor(
        input_channels=input_channels,
        hidden_channels=256,
        output_channels=128
    )

    spatial_feat, global_feat = shared_extractor(input_features)
    print(f"  空间特征: {spatial_feat.shape}")
    print(f"  全局特征: {global_feat.shape}")

    # 2. 测试任务特定头
    print("\n2. 测试任务特定头:")
    task_head = TaskSpecificHead(
        feature_dim=128,
        hidden_dim=64,
        task_name="T7d_M3-5",
        use_spatial_info=True
    )

    task_prediction = task_head(global_feat, spatial_feat)
    print(f"  任务预测: {task_prediction.shape}")
    print(f"  预测值范围: [{task_prediction.min():.3f}, {task_prediction.max():.3f}]")

    # 3. 测试完整多任务输出层
    print("\n3. 测试多任务输出层:")
    multi_task_layer = MultiTaskOutputLayer(
        input_channels=input_channels,
        shared_feature_dim=128,
        task_hidden_dim=64
    )

    # 计算参数数量
    total_params = sum(p.numel() for p in multi_task_layer.parameters())
    print(f"  模型参数数: {total_params:,}")

    # 前向传播
    with torch.no_grad():
        predictions = multi_task_layer(input_features)

    print(f"  所有任务预测: {predictions['all_tasks'].shape}")
    print(f"  预测值统计:")
    print(f"    最小值: {predictions['all_tasks'].min():.3f}")
    print(f"    最大值: {predictions['all_tasks'].max():.3f}")
    print(f"    平均值: {predictions['all_tasks'].mean():.3f}")

    # 4. 测试任务分组
    print("\n4. 测试任务分组:")
    all_preds = predictions['all_tasks']

    # 按时间窗口分组
    time_grouped = multi_task_layer.get_predictions_by_time_window(all_preds)
    print(f"  按时间窗口分组:")
    for time_window, preds in time_grouped.items():
        print(f"    {time_window}天: {preds.shape}")

    # 按震级分组
    mag_grouped = multi_task_layer.get_predictions_by_magnitude(all_preds)
    print(f"  按震级范围分组:")
    for mag_range, preds in mag_grouped.items():
        print(f"    M{mag_range[0]}-{mag_range[1]}: {preds.shape}")

    # 5. 测试预测后处理器
    print("\n5. 测试预测后处理器:")
    post_processor = PredictionPostProcessor(num_tasks=12)

    # 创建一些测试预测（故意违反一致性）
    test_predictions = torch.rand(batch_size, 12)

    # 应用后处理
    processed_predictions = post_processor(test_predictions)

    print(f"  原始预测: {test_predictions.shape}")
    print(f"  后处理预测: {processed_predictions.shape}")
    print(f"  一致性改进:")
    print(f"    原始范围: [{test_predictions.min():.3f}, {test_predictions.max():.3f}]")
    print(f"    处理后范围: [{processed_predictions.min():.3f}, {processed_predictions.max():.3f}]")

    # 6. 性能分析
    print("\n6. 性能分析:")
    input_memory = input_features.numel() * 4 / (1024**2)
    output_memory = all_preds.numel() * 4 / (1024**2)
    param_memory = total_params * 4 / (1024**2)

    print(f"  输入内存: {input_memory:.2f} MB")
    print(f"  输出内存: {output_memory:.2f} MB")
    print(f"  参数内存: {param_memory:.2f} MB")
    print(f"  总内存估算: {input_memory + output_memory + param_memory:.2f} MB")

    print("\n:white_check_mark: 多任务输出头模块测试完成!")
    return multi_task_layer, predictions, post_processor

if __name__ == "__main__":
    test_multi_task_heads()