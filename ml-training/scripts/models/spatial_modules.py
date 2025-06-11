"""
空间特征提取模块
实现多尺度CNN、空间金字塔池化、通道注意力等空间特征提取组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class ChannelAttention(nn.Module):
    """
    通道注意力模块 - 自适应加权不同特征通道
    基于SE-Net (Squeeze-and-Excitation Networks)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        初始化通道注意力
        
        Args:
            in_channels: 输入通道数
            reduction_ratio: 通道压缩比例
        """
        super(ChannelAttention, self).__init__()
        
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 通道压缩和恢复
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
        print(f"📊 通道注意力初始化: {in_channels}通道 → {in_channels//reduction_ratio} → {in_channels}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, channels, height, width]
            
        Returns:
            加权后的特征
        """
        batch, channels, height, width = x.size()
        
        # 全局信息提取 [batch, channels, 1, 1]
        global_info = self.global_avg_pool(x)
        
        # 通道重要性计算 [batch, channels]
        channel_weights = self.fc(global_info.view(batch, channels))
        
        # 重塑并应用权重
        channel_weights = channel_weights.view(batch, channels, 1, 1)
        
        return x * channel_weights

class SpatialAttention(nn.Module):
    """
    空间注意力模块 - 强调重要的空间位置
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        初始化空间注意力
        
        Args:
            kernel_size: 卷积核大小
        """
        super(SpatialAttention, self).__init__()
        
        self.kernel_size = kernel_size
        
        # 空间注意力卷积
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        print(f"🗺️  空间注意力初始化: 卷积核大小 {kernel_size}×{kernel_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, channels, height, width]
            
        Returns:
            加权后的特征
        """
        # 通道维度压缩：最大值和平均值
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [batch, 1, height, width]
        avg_pool = torch.mean(x, dim=1, keepdim=True)    # [batch, 1, height, width]
        
        # 拼接两个通道
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)  # [batch, 2, height, width]
        
        # 计算空间注意力权重
        spatial_weights = self.sigmoid(self.conv(spatial_input))  # [batch, 1, height, width]
        
        return x * spatial_weights

class MultiScaleConvBlock(nn.Module):
    """
    多尺度卷积块 - 并行提取不同尺度的空间特征
    类似Inception模块的设计
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化多尺度卷积块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super(MultiScaleConvBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 分配输出通道
        branch_channels = out_channels // 4
        
        # 分支1: 1×1卷积 (点特征)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支2: 3×3卷积 (局部特征)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支3: 5×5卷积 (区域特征)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支4: 最大池化 + 1×1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        print(f"🔀 多尺度卷积块: {in_channels} → {out_channels} (4分支×{branch_channels})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, in_channels, height, width]
            
        Returns:
            多尺度特征 [batch, out_channels, height, width]
        """
        # 并行计算四个分支
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        
        # 拼接所有分支
        output = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)
        
        return output

class SpatialPyramidPooling(nn.Module):
    """
    空间金字塔池化 - 捕捉多种空间尺度的信息
    适应不同大小的地震影响区域
    """
    
    def __init__(self, in_channels: int, pool_sizes: List[int] = [1, 2, 3, 6]):
        """
        初始化空间金字塔池化
        
        Args:
            in_channels: 输入通道数
            pool_sizes: 池化尺寸列表
        """
        super(SpatialPyramidPooling, self).__init__()
        
        self.pool_sizes = pool_sizes
        self.pooling_layers = nn.ModuleList()
        
        for pool_size in pool_sizes:
            self.pooling_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(in_channels, in_channels // len(pool_sizes), 
                             kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_channels // len(pool_sizes)),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 输出通道数
        self.out_channels = in_channels
        
        print(f"🏔️  空间金字塔池化: 池化尺寸 {pool_sizes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, channels, height, width]
            
        Returns:
            金字塔池化特征
        """
        batch, channels, height, width = x.size()
        pyramid_features = []
        
        for pooling_layer in self.pooling_layers:
            # 池化到不同尺寸
            pooled = pooling_layer(x)
            
            # 上采样回原始尺寸
            upsampled = F.interpolate(pooled, size=(height, width), 
                                    mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)
        
        # 拼接所有金字塔特征
        output = torch.cat(pyramid_features, dim=1)
        
        return output

class ResidualBlock(nn.Module):
    """
    残差块 - 解决深度网络梯度消失问题
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        初始化残差块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 卷积步长
        """
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class MultiScaleCNN(nn.Module):
    """
    多尺度CNN主模块 - 整合所有空间特征提取组件
    """
    
    def __init__(self, 
                 in_channels: int = 8,
                 base_channels: int = 64,
                 num_scales: int = 3):
        """
        初始化多尺度CNN
        
        Args:
            in_channels: 输入通道数 (8个特征通道)
            base_channels: 基础通道数
            num_scales: 尺度数量
        """
        super(MultiScaleCNN, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_scales = num_scales
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 保持空间尺寸
        )
        
        # 多尺度特征提取层
        self.multi_scale_blocks = nn.ModuleList()
        current_channels = base_channels
        
        for scale in range(num_scales):
            output_channels = base_channels * (2 ** scale)
            
            # 多尺度卷积块
            multi_scale_block = MultiScaleConvBlock(current_channels, output_channels)
            self.multi_scale_blocks.append(multi_scale_block)
            
            # 残差块增强特征
            residual_block = ResidualBlock(output_channels, output_channels)
            self.multi_scale_blocks.append(residual_block)
            
            current_channels = output_channels
        
        # 空间金字塔池化
        self.spatial_pyramid = SpatialPyramidPooling(current_channels)
        
        # 通道注意力
        self.channel_attention = ChannelAttention(current_channels)
        
        # 空间注意力
        self.spatial_attention = SpatialAttention()
        
        # 输出特征维度
        self.out_channels = current_channels
        
        print(f"🧠 多尺度CNN初始化:")
        print(f"  输入通道: {in_channels}")
        print(f"  基础通道: {base_channels}")
        print(f"  尺度数: {num_scales}")
        print(f"  输出通道: {self.out_channels}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, time_steps, height, width, channels]
            
        Returns:
            空间特征 [batch, time_steps, height', width', out_channels]
        """
        batch_size, time_steps, height, width, channels = x.size()
        
        # 重塑为2D卷积格式 [batch*time_steps, channels, height, width]
        x = x.view(batch_size * time_steps, channels, height, width)
        
        # 初始特征提取
        x = self.initial_conv(x)
        
        # 多尺度特征提取
        for block in self.multi_scale_blocks:
            x = block(x)
        
        # 空间金字塔池化
        x = self.spatial_pyramid(x)
        
        # 通道注意力
        x = self.channel_attention(x)
        
        # 空间注意力
        x = self.spatial_attention(x)
        
        # 重塑回时间序列格式
        _, out_channels, out_height, out_width = x.size()
        x = x.view(batch_size, time_steps, out_height, out_width, out_channels)
        
        return x
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        计算输出形状
        
        Args:
            input_shape: 输入形状 (time_steps, height, width, channels)
            
        Returns:
            输出形状
        """
        time_steps, height, width, channels = input_shape
        
        # 假设空间维度不变 (使用padding保持)
        output_shape = (time_steps, height, width, self.out_channels)
        
        return output_shape

def test_spatial_modules():
    """测试空间特征提取模块"""
    print("🧪 测试空间特征提取模块...")
    
    # 模拟输入数据 [batch, time_steps, height, width, channels]
    batch_size = 2
    time_steps = 90
    height, width = 22, 24
    in_channels = 8
    
    input_tensor = torch.randn(batch_size, time_steps, height, width, in_channels)
    print(f"📊 输入数据形状: {input_tensor.shape}")
    
    # 1. 测试通道注意力
    print("\n1. 测试通道注意力:")
    test_input_2d = torch.randn(4, 64, 22, 24)  # 2D输入用于测试
    channel_attn = ChannelAttention(64, reduction_ratio=16)
    channel_output = channel_attn(test_input_2d)
    print(f"  输入: {test_input_2d.shape} → 输出: {channel_output.shape}")
    
    # 2. 测试空间注意力
    print("\n2. 测试空间注意力:")
    spatial_attn = SpatialAttention(kernel_size=7)
    spatial_output = spatial_attn(test_input_2d)
    print(f"  输入: {test_input_2d.shape} → 输出: {spatial_output.shape}")
    
    # 3. 测试多尺度卷积块
    print("\n3. 测试多尺度卷积块:")
    multi_scale_block = MultiScaleConvBlock(in_channels=64, out_channels=128)
    ms_output = multi_scale_block(test_input_2d)
    print(f"  输入: {test_input_2d.shape} → 输出: {ms_output.shape}")
    
    # 4. 测试空间金字塔池化
    print("\n4. 测试空间金字塔池化:")
    spp = SpatialPyramidPooling(in_channels=64, pool_sizes=[1, 2, 3, 6])
    spp_output = spp(test_input_2d)
    print(f"  输入: {test_input_2d.shape} → 输出: {spp_output.shape}")
    
    # 5. 测试残差块
    print("\n5. 测试残差块:")
    residual_block = ResidualBlock(in_channels=64, out_channels=128, stride=1)
    res_output = residual_block(test_input_2d)
    print(f"  输入: {test_input_2d.shape} → 输出: {res_output.shape}")
    
    # 6. 测试完整多尺度CNN
    print("\n6. 测试完整多尺度CNN:")
    multi_scale_cnn = MultiScaleCNN(
        in_channels=in_channels,
        base_channels=64,
        num_scales=3
    )
    
    # 计算模型参数
    total_params = sum(p.numel() for p in multi_scale_cnn.parameters())
    print(f"  模型参数数: {total_params:,}")
    
    # 前向传播测试
    with torch.no_grad():
        output = multi_scale_cnn(input_tensor)
        print(f"  输入: {input_tensor.shape}")
        print(f"  输出: {output.shape}")
    
    # 7. 输出形状预测
    print("\n7. 输出形状预测:")
    predicted_shape = multi_scale_cnn.get_output_shape((time_steps, height, width, in_channels))
    print(f"  预测输出形状: {predicted_shape}")
    print(f"  实际输出形状: {tuple(output.shape[1:])}")
    
    # 8. 内存使用估算
    print("\n8. 内存使用估算:")
    input_memory = input_tensor.numel() * 4 / (1024**2)  # MB
    output_memory = output.numel() * 4 / (1024**2)  # MB
    print(f"  输入内存: {input_memory:.2f} MB")
    print(f"  输出内存: {output_memory:.2f} MB")
    
    print("\n✅ 空间特征提取模块测试完成!")
    return multi_scale_cnn, output

if __name__ == "__main__":
    test_spatial_modules()