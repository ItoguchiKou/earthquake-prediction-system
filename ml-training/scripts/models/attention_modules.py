"""
注意力机制模块
实现多头注意力、跨模态注意力、空间-时间注意力等高级注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Union
import warnings
warnings.filterwarnings('ignore')

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制 - Transformer风格的注意力
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 bias: bool = True):
        """
        初始化多头注意力
        
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout比例
            bias: 是否使用偏置
        """
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K, V投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"🧠 多头注意力初始化: {embed_dim}维, {num_heads}头, 每头{self.head_dim}维")
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch, seq_len, embed_dim]
            key: 键张量 [batch, seq_len, embed_dim]
            value: 值张量 [batch, seq_len, embed_dim]
            mask: 注意力掩码 [batch, seq_len, seq_len] (可选)
            
        Returns:
            (注意力输出, 注意力权重)
        """
        batch_size, seq_len, embed_dim = query.size()
        
        # 线性投影
        Q = self.q_proj(query)  # [batch, seq_len, embed_dim]
        K = self.k_proj(key)    # [batch, seq_len, embed_dim]
        V = self.v_proj(value)  # [batch, seq_len, embed_dim]
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在形状: [batch, num_heads, seq_len, head_dim]
        
        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # 形状: [batch, num_heads, seq_len, seq_len]
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores.masked_fill_(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attention_output = torch.matmul(attention_weights, V)
        # 形状: [batch, num_heads, seq_len, head_dim]
        
        # 重塑回原始格式
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # 输出投影
        output = self.out_proj(attention_output)
        
        # 返回平均注意力权重 (所有头的平均)
        avg_attention_weights = attention_weights.mean(dim=1)
        
        return output, avg_attention_weights

class SpatioTemporalAttention(nn.Module):
    """
    时空注意力 - 同时建模空间和时间维度的注意力
    """
    
    def __init__(self,
                 spatial_dim: int,
                 temporal_dim: int,
                 hidden_dim: int = 128):
        """
        初始化时空注意力
        
        Args:
            spatial_dim: 空间特征维度
            temporal_dim: 时间特征维度  
            hidden_dim: 隐藏层维度
        """
        super(SpatioTemporalAttention, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        
        # 空间注意力分支
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(spatial_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 时间注意力分支
        self.temporal_attention = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 交互注意力 (空间-时间)
        self.interaction_attention = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        print(f"🌐 时空注意力初始化: 空间{spatial_dim}维 + 时间{temporal_dim}维 → {hidden_dim}维")
    
    def forward(self, 
                spatial_features: torch.Tensor,
                temporal_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            spatial_features: 空间特征 [batch, time_steps, channels, height, width]
            temporal_features: 时间特征 [batch, time_steps, channels, height, width]
            
        Returns:
            (融合特征, 注意力权重字典)
        """
        batch_size, time_steps, channels, height, width = spatial_features.size()
        
        # 1. 空间注意力计算
        spatial_weights = []
        for t in range(time_steps):
            # 计算该时间步的空间注意力
            spatial_weight = self.spatial_attention(spatial_features[:, t, :, :, :])
            spatial_weights.append(spatial_weight)
        
        spatial_attention_weights = torch.stack(spatial_weights, dim=1)
        # 形状: [batch, time_steps, 1, height, width]
        
        # 2. 时间注意力计算
        # 将空间维度压缩为特征向量
        spatial_pooled = F.adaptive_avg_pool2d(
            spatial_features.view(batch_size * time_steps, channels, height, width),
            (1, 1)
        ).view(batch_size, time_steps, channels)
        
        temporal_pooled = F.adaptive_avg_pool2d(
            temporal_features.view(batch_size * time_steps, channels, height, width),
            (1, 1)
        ).view(batch_size, time_steps, channels)
        
        # 计算时间注意力权重
        temporal_attention_weights = self.temporal_attention(temporal_pooled)
        # 形状: [batch, time_steps, 1]
        
        # 3. 交互注意力计算
        combined_features = torch.cat([spatial_pooled, temporal_pooled], dim=-1)
        interaction_weights = self.interaction_attention(combined_features)
        # 形状: [batch, time_steps, hidden_dim]
        
        # 4. 应用注意力权重
        # 空间注意力 - 直接广播，因为维度已经匹配
        spatial_attended = spatial_features * spatial_attention_weights
        
        # 时间注意力 - 需要正确扩展维度到 [batch, time_steps, 1, 1, 1]
        temporal_weight_expanded = temporal_attention_weights.unsqueeze(-1).unsqueeze(-1)
        # 现在形状: [batch, time_steps, 1, 1, 1]
        temporal_attended = temporal_features * temporal_weight_expanded
        
        # 特征融合
        fused_features = spatial_attended + temporal_attended
        
        # 交互权重应用 - 平均后扩展维度
        interaction_weight_avg = interaction_weights.mean(dim=-1, keepdim=True)  # [batch, time_steps, 1]
        interaction_weight_expanded = interaction_weight_avg.unsqueeze(-1).unsqueeze(-1)
        # 现在形状: [batch, time_steps, 1, 1, 1]
        fused_features = fused_features * interaction_weight_expanded
        
        attention_weights = {
            'spatial': spatial_attention_weights.squeeze(2),  # [batch, time_steps, height, width]
            'temporal': temporal_attention_weights.squeeze(-1),  # [batch, time_steps]
            'interaction': interaction_weights  # [batch, time_steps, hidden_dim]
        }
        
        return fused_features, attention_weights

class CrossModalAttention(nn.Module):
    """
    跨模态注意力 - 不同特征模态之间的注意力交互
    """
    
    def __init__(self,
                 modality1_dim: int,
                 modality2_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8):
        """
        初始化跨模态注意力
        
        Args:
            modality1_dim: 模态1特征维度
            modality2_dim: 模态2特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
        """
        super(CrossModalAttention, self).__init__()
        
        self.modality1_dim = modality1_dim
        self.modality2_dim = modality2_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 模态1 → 模态2 注意力
        self.m1_to_m2_attention = MultiHeadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # 模态2 → 模态1 注意力
        self.m2_to_m1_attention = MultiHeadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # 投影层
        self.m1_proj = nn.Linear(modality1_dim, hidden_dim)
        self.m2_proj = nn.Linear(modality2_dim, hidden_dim)
        
        # 输出投影
        self.m1_out_proj = nn.Linear(hidden_dim, modality1_dim)
        self.m2_out_proj = nn.Linear(hidden_dim, modality2_dim)
        
        print(f"🔄 跨模态注意力初始化: {modality1_dim}维 ↔ {modality2_dim}维")
    
    def forward(self,
                modality1_features: torch.Tensor,
                modality2_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            modality1_features: 模态1特征 [batch, seq_len, modality1_dim]
            modality2_features: 模态2特征 [batch, seq_len, modality2_dim]
            
        Returns:
            (增强的模态1特征, 增强的模态2特征, 注意力权重)
        """
        # 投影到统一维度
        m1_projected = self.m1_proj(modality1_features)
        m2_projected = self.m2_proj(modality2_features)
        
        # 模态1 → 模态2 注意力
        m1_enhanced, m1_to_m2_weights = self.m1_to_m2_attention(
            query=m1_projected,
            key=m2_projected,
            value=m2_projected
        )
        
        # 模态2 → 模态1 注意力
        m2_enhanced, m2_to_m1_weights = self.m2_to_m1_attention(
            query=m2_projected,
            key=m1_projected,
            value=m1_projected
        )
        
        # 残差连接
        m1_enhanced = m1_enhanced + m1_projected
        m2_enhanced = m2_enhanced + m2_projected
        
        # 投影回原始维度
        m1_output = self.m1_out_proj(m1_enhanced)
        m2_output = self.m2_out_proj(m2_enhanced)
        
        # 残差连接到原始特征
        m1_output = m1_output + modality1_features
        m2_output = m2_output + modality2_features
        
        attention_weights = {
            'm1_to_m2': m1_to_m2_weights,
            'm2_to_m1': m2_to_m1_weights
        }
        
        return m1_output, m2_output, attention_weights

class GlobalLocalAttention(nn.Module):
    """
    全局-局部注意力 - 结合全局上下文和局部细节
    """
    
    def __init__(self,
                 feature_dim: int,
                 num_scales: int = 3,
                 hidden_dim: int = 128):
        """
        初始化全局-局部注意力
        
        Args:
            feature_dim: 特征维度
            num_scales: 尺度数量
            hidden_dim: 隐藏层维度
        """
        super(GlobalLocalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        
        # 全局注意力 (全局平均池化)
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 多尺度局部注意力
        self.local_attentions = nn.ModuleList()
        for scale in range(num_scales):
            kernel_size = 2 ** (scale + 1) + 1  # 3, 5, 9
            padding = kernel_size // 2
            
            local_attn = nn.Sequential(
                nn.Conv2d(feature_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
                nn.Sigmoid()
            )
            self.local_attentions.append(local_attn)
        
        # 尺度融合
        self.scale_fusion = nn.Conv2d(num_scales, 1, kernel_size=1)
        
        print(f"🌍 全局-局部注意力初始化: {feature_dim}维, {num_scales}个尺度")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, channels, height, width]
            
        Returns:
            (注意力增强特征, 注意力权重)
        """
        # 全局注意力
        global_weights = self.global_attention(x)
        
        # 多尺度局部注意力
        local_weights_list = []
        for local_attention in self.local_attentions:
            local_weight = local_attention(x)
            local_weights_list.append(local_weight)
        
        # 融合多尺度局部注意力
        local_weights_stacked = torch.cat(local_weights_list, dim=1)
        fused_local_weights = self.scale_fusion(local_weights_stacked)
        
        # 组合全局和局部注意力
        combined_weights = global_weights * fused_local_weights
        
        # 应用注意力权重
        attended_features = x * combined_weights
        
        attention_weights = {
            'global': global_weights,
            'local_scales': local_weights_list,
            'fused_local': fused_local_weights,
            'combined': combined_weights
        }
        
        return attended_features, attention_weights

class EarthquakeAttentionFusion(nn.Module):
    """
    地震预测专用注意力融合模块
    整合多种注意力机制用于地震特征增强
    """
    
    def __init__(self,
                 spatial_channels: int = 256,
                 temporal_channels: int = 256,
                 fusion_dim: int = 512,
                 num_heads: int = 8):
        """
        初始化地震注意力融合
        
        Args:
            spatial_channels: 空间特征通道数
            temporal_channels: 时间特征通道数
            fusion_dim: 融合后特征维度
            num_heads: 多头注意力头数
        """
        super(EarthquakeAttentionFusion, self).__init__()
        
        self.spatial_channels = spatial_channels
        self.temporal_channels = temporal_channels
        self.fusion_dim = fusion_dim
        
        # 时空注意力
        self.spatiotemporal_attention = SpatioTemporalAttention(
            spatial_dim=spatial_channels,
            temporal_dim=temporal_channels,
            hidden_dim=128
        )
        
        # 全局-局部注意力
        self.global_local_attention = GlobalLocalAttention(
            feature_dim=spatial_channels,
            num_scales=3,
            hidden_dim=64
        )
        
        # 特征融合投影
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(spatial_channels + temporal_channels, fusion_dim, kernel_size=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # 自注意力增强
        self.self_attention_2d = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim // 8, fusion_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        print(f"🌍 地震注意力融合初始化:")
        print(f"  空间通道: {spatial_channels}")
        print(f"  时间通道: {temporal_channels}")
        print(f"  融合维度: {fusion_dim}")
        print(f"  注意力头数: {num_heads}")
    
    def forward(self,
                spatial_features: torch.Tensor,
                temporal_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            spatial_features: 空间特征 [batch, time_steps, spatial_channels, height, width]
            temporal_features: 时间特征 [batch, time_steps, temporal_channels, height, width]
            
        Returns:
            (融合特征, 所有注意力权重)
        """
        batch_size, time_steps, _, height, width = spatial_features.size()
        
        # 1. 时空注意力增强
        st_features, st_attention = self.spatiotemporal_attention(
            spatial_features, temporal_features
        )
        
        # 2. 全局-局部注意力 (对每个时间步应用)
        gl_features = []
        gl_attentions = []
        
        for t in range(time_steps):
            gl_feat, gl_attn = self.global_local_attention(st_features[:, t, :, :, :])
            gl_features.append(gl_feat)
            gl_attentions.append(gl_attn)
        
        gl_features = torch.stack(gl_features, dim=1)
        
        # 3. 特征融合
        # 拼接空间和时间特征
        combined_features = torch.cat([spatial_features, temporal_features], dim=2)
        
        # 逐帧融合投影
        fused_frames = []
        for t in range(time_steps):
            fused_frame = self.fusion_proj(combined_features[:, t, :, :, :])
            fused_frames.append(fused_frame)
        
        fused_features = torch.stack(fused_frames, dim=1)
        
        # 4. 自注意力增强
        self_attended_frames = []
        for t in range(time_steps):
            self_attn_weight = self.self_attention_2d(fused_features[:, t, :, :, :])
            self_attended_frame = fused_features[:, t, :, :, :] * self_attn_weight
            self_attended_frames.append(self_attended_frame)
        
        final_features = torch.stack(self_attended_frames, dim=1)
        
        # 5. 残差连接 (如果维度匹配)
        if gl_features.size(2) == final_features.size(2):
            final_features = final_features + gl_features
        
        # 收集所有注意力权重
        all_attention_weights = {
            'spatiotemporal': st_attention,
            'global_local': gl_attentions[0] if gl_attentions else None,  # 只返回第一帧的示例
            'self_attention': self_attended_frames  # 自注意力权重包含在特征中
        }
        
        return final_features, all_attention_weights

def test_attention_modules():
    """测试注意力机制模块"""
    print("🧪 测试注意力机制模块...")
    
    # 模拟数据
    batch_size = 2
    seq_len = 10
    height, width = 22, 24
    embed_dim = 256
    
    # 1. 测试多头注意力
    print("\n1. 测试多头注意力:")
    multi_head_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=8)
    
    # 创建序列数据 [batch, seq_len, embed_dim]
    sequence_data = torch.randn(batch_size, seq_len, embed_dim)
    attn_output, attn_weights = multi_head_attn(sequence_data, sequence_data, sequence_data)
    
    print(f"  输入: {sequence_data.shape}")
    print(f"  输出: {attn_output.shape}")
    print(f"  注意力权重: {attn_weights.shape}")
    
    # 2. 测试时空注意力
    print("\n2. 测试时空注意力:")
    spatial_features = torch.randn(batch_size, seq_len, 128, height, width)
    temporal_features = torch.randn(batch_size, seq_len, 128, height, width)
    
    st_attention = SpatioTemporalAttention(spatial_dim=128, temporal_dim=128, hidden_dim=64)
    st_output, st_weights = st_attention(spatial_features, temporal_features)
    
    print(f"  空间特征: {spatial_features.shape}")
    print(f"  时间特征: {temporal_features.shape}")
    print(f"  融合输出: {st_output.shape}")
    print(f"  空间注意力: {st_weights['spatial'].shape}")
    print(f"  时间注意力: {st_weights['temporal'].shape}")
    
    # 3. 测试跨模态注意力
    print("\n3. 测试跨模态注意力:")
    modality1 = torch.randn(batch_size, seq_len, 256)
    modality2 = torch.randn(batch_size, seq_len, 128)
    
    cross_modal_attn = CrossModalAttention(modality1_dim=256, modality2_dim=128, hidden_dim=256)
    m1_enhanced, m2_enhanced, cm_weights = cross_modal_attn(modality1, modality2)
    
    print(f"  模态1输入: {modality1.shape} → 输出: {m1_enhanced.shape}")
    print(f"  模态2输入: {modality2.shape} → 输出: {m2_enhanced.shape}")
    
    # 4. 测试全局-局部注意力
    print("\n4. 测试全局-局部注意力:")
    feature_map = torch.randn(batch_size, 256, height, width)
    
    gl_attention = GlobalLocalAttention(feature_dim=256, num_scales=3)
    gl_output, gl_weights = gl_attention(feature_map)
    
    print(f"  输入特征图: {feature_map.shape}")
    print(f"  输出特征图: {gl_output.shape}")
    print(f"  全局注意力: {gl_weights['global'].shape}")
    print(f"  局部尺度数: {len(gl_weights['local_scales'])}")
    
    # 5. 测试完整地震注意力融合
    print("\n5. 测试地震注意力融合:")
    spatial_seq = torch.randn(batch_size, seq_len, 256, height, width)
    temporal_seq = torch.randn(batch_size, seq_len, 256, height, width)
    
    earthquake_fusion = EarthquakeAttentionFusion(
        spatial_channels=256,
        temporal_channels=256,
        fusion_dim=512,
        num_heads=8
    )
    
    # 计算参数数量
    total_params = sum(p.numel() for p in earthquake_fusion.parameters())
    print(f"  模型参数数: {total_params:,}")
    
    # 前向传播
    with torch.no_grad():
        fusion_output, all_weights = earthquake_fusion(spatial_seq, temporal_seq)
    
    print(f"  空间输入: {spatial_seq.shape}")
    print(f"  时间输入: {temporal_seq.shape}")
    print(f"  融合输出: {fusion_output.shape}")
    print(f"  注意力类型数: {len(all_weights)}")
    
    # 6. 性能分析
    print("\n6. 性能分析:")
    input_memory = (spatial_seq.numel() + temporal_seq.numel()) * 4 / (1024**2)
    output_memory = fusion_output.numel() * 4 / (1024**2)
    param_memory = total_params * 4 / (1024**2)
    
    print(f"  输入内存: {input_memory:.2f} MB")
    print(f"  输出内存: {output_memory:.2f} MB")
    print(f"  参数内存: {param_memory:.2f} MB")
    print(f"  总内存估算: {input_memory + output_memory + param_memory:.2f} MB")
    
    print("\n✅ 注意力机制模块测试完成!")
    return earthquake_fusion, fusion_output, all_weights

if __name__ == "__main__":
    test_attention_modules()