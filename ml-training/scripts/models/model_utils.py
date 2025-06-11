"""
模型工具函数模块
提供权重初始化、模型保存/加载、可视化等通用工具函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class ModelUtils:
    """模型工具类 - 提供各种实用功能"""
    
    @staticmethod
    def get_device():
        """获取可用设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 使用GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("💻 使用CPU")
        return device
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        统计模型参数数量
        
        Args:
            model: PyTorch模型
            
        Returns:
            参数统计字典
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    @staticmethod
    def print_model_info(model: nn.Module, input_shape: Tuple[int, ...]):
        """
        打印模型详细信息
        
        Args:
            model: PyTorch模型
            input_shape: 输入数据形状 (不包含batch_size)
        """
        print("\n" + "="*60)
        print("🧠 模型架构信息")
        print("="*60)
        
        # 参数统计
        param_stats = ModelUtils.count_parameters(model)
        print(f"📊 参数统计:")
        print(f"  总参数数: {param_stats['total_parameters']:,}")
        print(f"  可训练参数: {param_stats['trainable_parameters']:,}")
        print(f"  固定参数: {param_stats['non_trainable_parameters']:,}")
        
        # 模型大小估算
        param_size_mb = param_stats['total_parameters'] * 4 / (1024 * 1024)  # float32
        print(f"  模型大小: {param_size_mb:.2f} MB")
        
        # 输入输出形状
        print(f"\n🔄 输入输出信息:")
        print(f"  输入形状: {input_shape}")
        
        # 尝试推断输出形状
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, *input_shape)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                    model = model.cuda()
                output = model(dummy_input)
                if isinstance(output, torch.Tensor):
                    print(f"  输出形状: {tuple(output.shape[1:])}")
                elif isinstance(output, (list, tuple)):
                    for i, out in enumerate(output):
                        print(f"  输出{i}形状: {tuple(out.shape[1:])}")
        except Exception as e:
            print(f"  输出形状: 无法推断 ({e})")
        
        print("="*60)

def init_weights(module: nn.Module, init_type: str = 'xavier_uniform'):
    """
    权重初始化函数
    
    Args:
        module: 网络模块
        init_type: 初始化类型 ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(module.weight)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
        elif init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    
    elif isinstance(module, (nn.Linear)):
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(module.weight)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
        elif init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    
    elif isinstance(module, (nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if 'weight' in name:
                if init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(param)
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(param)
                elif 'kaiming' in init_type:
                    nn.init.kaiming_uniform_(param, mode='fan_out', nonlinearity='tanh')
            elif 'bias' in name:
                nn.init.constant_(param, 0)

def save_checkpoint(model: nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   epoch: int,
                   loss: float,
                   metrics: Dict[str, float],
                   save_path: str,
                   is_best: bool = False):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前轮次
        loss: 损失值
        metrics: 评估指标
        save_path: 保存路径
        is_best: 是否是最佳模型
    """
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 准备检查点数据
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存检查点
    torch.save(checkpoint, save_path)
    
    # 如果是最佳模型，额外保存一份
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"💾 最佳模型已保存: {best_path}")
    
    print(f"💾 检查点已保存: {save_path}")

def load_checkpoint(model: nn.Module,
                   checkpoint_path: str,
                   optimizer: torch.optim.Optimizer = None,
                   scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                   device: torch.device = None) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        optimizer: 优化器(可选)
        scheduler: 学习率调度器(可选)
        device: 设备
        
    Returns:
        检查点信息字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 加载检查点
    if device is None:
        device = ModelUtils.get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"📂 检查点已加载: {checkpoint_path}")
    print(f"   轮次: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   损失: {checkpoint.get('loss', 'Unknown')}")
    
    return checkpoint

def save_model_architecture(model: nn.Module, save_path: str):
    """
    保存模型架构信息
    
    Args:
        model: 模型
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 收集模型信息
    param_stats = ModelUtils.count_parameters(model)
    
    model_info = {
        'model_class': model.__class__.__name__,
        'model_structure': str(model),
        'parameter_stats': param_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存为JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"📋 模型架构已保存: {save_path}")

class AttentionVisualizer:
    """注意力权重可视化工具"""
    
    @staticmethod
    def plot_temporal_attention(attention_weights: np.ndarray,
                              time_labels: List[str] = None,
                              title: str = "时间注意力权重",
                              save_path: str = None):
        """
        可视化时间注意力权重
        
        Args:
            attention_weights: 注意力权重 [time_steps] 或 [batch, time_steps]
            time_labels: 时间标签
            title: 图表标题
            save_path: 保存路径
        """
        if attention_weights.ndim > 1:
            attention_weights = attention_weights.mean(axis=0)  # 平均多个样本
        
        plt.figure(figsize=(12, 6))
        
        time_steps = len(attention_weights)
        x_pos = np.arange(time_steps)
        
        # 绘制柱状图
        bars = plt.bar(x_pos, attention_weights, alpha=0.7, color='skyblue', edgecolor='navy')
        
        # 突出显示高权重区域
        max_weight = np.max(attention_weights)
        for i, (bar, weight) in enumerate(zip(bars, attention_weights)):
            if weight > 0.8 * max_weight:
                bar.set_color('orange')
        
        # 设置标签
        if time_labels:
            plt.xticks(x_pos[::5], time_labels[::5], rotation=45)
        else:
            plt.xticks(x_pos[::10], [f"T-{time_steps-i}" for i in x_pos[::10]])
        
        plt.xlabel('时间步')
        plt.ylabel('注意力权重')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 时间注意力图已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_spatial_attention(attention_weights: np.ndarray,
                             title: str = "空间注意力权重",
                             save_path: str = None):
        """
        可视化空间注意力权重
        
        Args:
            attention_weights: 空间注意力权重 [height, width] 或 [batch, height, width]
            title: 图表标题
            save_path: 保存路径
        """
        if attention_weights.ndim > 2:
            attention_weights = attention_weights.mean(axis=0)  # 平均多个样本
        
        plt.figure(figsize=(10, 8))
        
        # 绘制热力图
        sns.heatmap(attention_weights, 
                   cmap='YlOrRd', 
                   annot=False,
                   cbar_kws={'label': '注意力权重'},
                   xticklabels=False,
                   yticklabels=False)
        
        plt.title(title)
        plt.xlabel('经度网格')
        plt.ylabel('纬度网格')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 空间注意力图已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_channel_attention(attention_weights: np.ndarray,
                             channel_names: List[str] = None,
                             title: str = "通道注意力权重",
                             save_path: str = None):
        """
        可视化通道注意力权重
        
        Args:
            attention_weights: 通道注意力权重 [channels] 或 [batch, channels]
            channel_names: 通道名称
            title: 图表标题
            save_path: 保存路径
        """
        if attention_weights.ndim > 1:
            attention_weights = attention_weights.mean(axis=0)  # 平均多个样本
        
        plt.figure(figsize=(10, 6))
        
        channels = len(attention_weights)
        x_pos = np.arange(channels)
        
        # 绘制柱状图
        bars = plt.bar(x_pos, attention_weights, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        
        # 突出显示重要通道
        max_weight = np.max(attention_weights)
        for i, (bar, weight) in enumerate(zip(bars, attention_weights)):
            if weight > 0.8 * max_weight:
                bar.set_color('red')
        
        # 设置标签
        if channel_names:
            plt.xticks(x_pos, channel_names, rotation=45, ha='right')
        else:
            plt.xticks(x_pos, [f"通道{i}" for i in range(channels)])
        
        plt.xlabel('特征通道')
        plt.ylabel('注意力权重')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 通道注意力图已保存: {save_path}")
        
        plt.show()

def calculate_model_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    估算模型FLOPs (浮点运算次数)
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状 (不包含batch_size)
        
    Returns:
        估算的FLOPs数量
    """
    try:
        from thop import profile
        
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        return int(flops)
    except ImportError:
        print("⚠️  需要安装thop库来计算FLOPs: pip install thop")
        return 0
    except Exception as e:
        print(f"⚠️  FLOPs计算失败: {e}")
        return 0

def get_model_memory_usage(model: nn.Module, input_shape: Tuple[int, ...], batch_size: int = 1) -> Dict[str, float]:
    """
    估算模型内存使用量
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状
        batch_size: 批次大小
        
    Returns:
        内存使用量字典 (MB)
    """
    # 参数内存
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    # 输入内存
    input_memory = np.prod([batch_size] + list(input_shape)) * 4 / (1024 * 1024)  # float32
    
    # 估算激活内存 (粗略估计为参数的2-4倍)
    activation_memory = param_memory * 3
    
    # 梯度内存 (训练时，约等于参数内存)
    gradient_memory = param_memory
    
    return {
        'parameters_mb': param_memory,
        'input_mb': input_memory,
        'activations_mb': activation_memory,
        'gradients_mb': gradient_memory,
        'total_training_mb': param_memory + input_memory + activation_memory + gradient_memory,
        'total_inference_mb': param_memory + input_memory + activation_memory
    }

def test_model_utils():
    """测试模型工具函数"""
    print("🧪 测试模型工具函数...")
    
    # 创建简单测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 22 * 24, 12)
        
        def forward(self, x):
            # x shape: [B, 90, 22, 24, 8] -> [B*90, 8, 22, 24]
            B, T, H, W, C = x.shape
            x = x.view(B*T, C, H, W)
            x = F.relu(self.conv(x))
            x = x.view(B*T, -1)
            x = self.fc(x)
            x = x.view(B, T, -1)
            return x.mean(dim=1)  # [B, 12]
    
    model = SimpleModel()
    input_shape = (90, 22, 24, 8)
    
    # 测试各种功能
    print("\n1. 模型信息:")
    ModelUtils.print_model_info(model, input_shape)
    
    print("\n2. 权重初始化:")
    model.apply(lambda m: init_weights(m, 'xavier_uniform'))
    print("✅ 权重初始化完成")
    
    print("\n3. 内存使用估算:")
    memory_usage = get_model_memory_usage(model, input_shape, batch_size=4)
    for key, value in memory_usage.items():
        print(f"  {key}: {value:.2f} MB")
    
    print("\n4. 注意力可视化测试:")
    # 模拟注意力权重 (修复softmax问题)
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    temporal_weights = softmax(np.random.randn(90))
    spatial_weights = softmax(np.random.randn(22, 24).flatten()).reshape(22, 24)
    channel_weights = softmax(np.random.randn(8))
    
    channel_names = [
        "地震频率", "平均震级", "最大震级", "能量释放",
        "震级标准差", "平均深度", "时间密度", "空间相关性"
    ]
    
    # 注意: 在实际使用中会显示图像，这里只测试函数调用
    print("  📊 时间注意力权重形状:", temporal_weights.shape)
    print("  📊 空间注意力权重形状:", spatial_weights.shape)
    print("  📊 通道注意力权重形状:", channel_weights.shape)
    
    print("\n✅ 模型工具函数测试完成!")

if __name__ == "__main__":
    test_model_utils()