"""
时间序列特征提取模块
实现ConvLSTM、时间注意力、双向LSTM等时序建模组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class ConvLSTMCell(nn.Module):
    """
    卷积LSTM单元 - 结合卷积和LSTM的时空建模能力
    """
    
    def __init__(self, 
                 input_channels: int,
                 hidden_channels: int,
                 kernel_size: int = 3,
                 bias: bool = True):
        """
        初始化ConvLSTM单元
        
        Args:
            input_channels: 输入通道数
            hidden_channels: 隐藏状态通道数
            kernel_size: 卷积核大小
            bias: 是否使用偏置
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # 输入到隐藏状态的卷积 (Input-to-Hidden)
        self.conv_ih = nn.Conv2d(
            input_channels, 4 * hidden_channels,
            kernel_size, padding=self.padding, bias=bias
        )
        
        # 隐藏状态到隐藏状态的卷积 (Hidden-to-Hidden)
        self.conv_hh = nn.Conv2d(
            hidden_channels, 4 * hidden_channels,
            kernel_size, padding=self.padding, bias=bias
        )
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, input_tensor: torch.Tensor, 
                hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_tensor: 输入张量 [batch, channels, height, width]
            hidden_state: (h_t, c_t) 隐藏状态和细胞状态
            
        Returns:
            新的隐藏状态 (h_t+1, c_t+1)
        """
        h_prev, c_prev = hidden_state
        
        # 输入变换
        conv_input = self.conv_ih(input_tensor)
        
        # 隐藏状态变换
        conv_hidden = self.conv_hh(h_prev)
        
        # 组合输入和隐藏状态
        conv_combined = conv_input + conv_hidden
        
        # 分离四个门
        i, f, o, g = torch.split(conv_combined, self.hidden_channels, dim=1)
        
        # 计算门值
        input_gate = torch.sigmoid(i)      # 输入门
        forget_gate = torch.sigmoid(f)     # 遗忘门
        output_gate = torch.sigmoid(o)     # 输出门
        candidate = torch.tanh(g)          # 候选值
        
        # 更新细胞状态
        c_next = forget_gate * c_prev + input_gate * candidate
        
        # 更新隐藏状态
        h_next = output_gate * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, height: int, width: int, device: torch.device):
        """
        初始化隐藏状态
        
        Args:
            batch_size: 批次大小
            height: 特征图高度
            width: 特征图宽度
            device: 设备
            
        Returns:
            初始隐藏状态 (h_0, c_0)
        """
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c

class ConvLSTM(nn.Module):
    """
    多层卷积LSTM - 处理时空序列数据
    """
    
    def __init__(self,
                 input_channels: int,
                 hidden_channels: List[int],
                 kernel_sizes: List[int],
                 num_layers: int,
                 bias: bool = True,
                 return_all_layers: bool = False):
        """
        初始化ConvLSTM
        
        Args:
            input_channels: 输入通道数
            hidden_channels: 各层隐藏通道数列表
            kernel_sizes: 各层卷积核大小列表
            num_layers: 层数
            bias: 是否使用偏置
            return_all_layers: 是否返回所有层的输出
        """
        super(ConvLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        # 构建各层
        cell_list = []
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels[i-1]
            cell_list.append(ConvLSTMCell(
                input_channels=cur_input_channels,
                hidden_channels=hidden_channels[i],
                kernel_size=kernel_sizes[i],
                bias=bias
            ))
        
        self.cell_list = nn.ModuleList(cell_list)
        
        print(f"⏰ ConvLSTM初始化:")
        print(f"  层数: {num_layers}")
        print(f"  隐藏通道: {hidden_channels}")
        print(f"  卷积核大小: {kernel_sizes}")
    
    def forward(self, input_tensor: torch.Tensor, 
                hidden_state: Optional[List] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        前向传播
        
        Args:
            input_tensor: 输入序列 [batch, time_steps, channels, height, width]
            hidden_state: 初始隐藏状态
            
        Returns:
            (layer_output_list, last_state_list)
        """
        batch_size, seq_len, _, height, width = input_tensor.size()
        device = input_tensor.device
        
        # 初始化隐藏状态
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width, device)
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size: int, height: int, width: int, device: torch.device):
        """初始化所有层的隐藏状态"""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, height, width, device))
        return init_states

class TemporalAttention(nn.Module):
    """
    时间注意力模块 - 识别重要的时间步
    """
    
    def __init__(self, 
                 input_channels: int,
                 attention_dim: int = 128,
                 use_position_encoding: bool = True):
        """
        初始化时间注意力
        
        Args:
            input_channels: 输入通道数
            attention_dim: 注意力维度
            use_position_encoding: 是否使用位置编码
        """
        super(TemporalAttention, self).__init__()
        
        self.input_channels = input_channels
        self.attention_dim = attention_dim
        self.use_position_encoding = use_position_encoding
        
        # 注意力计算网络
        self.attention_conv = nn.Sequential(
            nn.Conv2d(input_channels, attention_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, 1, kernel_size=1)
        )
        
        # 位置编码
        if use_position_encoding:
            self.position_encoding = PositionalEncoding(input_channels)
        
        print(f"⏰ 时间注意力初始化: {input_channels} → {attention_dim} → 1")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch, time_steps, channels, height, width]
            
        Returns:
            (加权序列, 注意力权重)
        """
        batch_size, time_steps, channels, height, width = x.size()
        
        # 添加位置编码
        if self.use_position_encoding:
            x = self.position_encoding(x)
        
        # 计算每个时间步的注意力分数
        attention_scores = []
        
        for t in range(time_steps):
            # 计算该时间步的注意力分数 [batch, 1, height, width]
            score = self.attention_conv(x[:, t, :, :, :])
            attention_scores.append(score)
        
        # 堆叠并计算softmax [batch, time_steps, height, width]
        attention_scores = torch.stack(attention_scores, dim=1).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 应用注意力权重
        weighted_sequence = []
        for t in range(time_steps):
            # 获取当前时间步的权重 [batch, height, width]
            weight = attention_weights[:, t, :, :]  # [batch, height, width]
            # 扩展维度以匹配特征 [batch, 1, height, width]
            weight = weight.unsqueeze(1)
            
            # 应用权重到当前时间步的特征
            weighted_frame = x[:, t, :, :, :] * weight  # 广播乘法
            weighted_sequence.append(weighted_frame)
        
        weighted_sequence = torch.stack(weighted_sequence, dim=1)
        
        return weighted_sequence, attention_weights

class PositionalEncoding(nn.Module):
    """
    位置编码 - 为时间序列添加位置信息
    """
    
    def __init__(self, channels: int, max_len: int = 200):
        """
        初始化位置编码
        
        Args:
            channels: 通道数
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        self.channels = channels
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, channels, 2).float() * 
                           (-math.log(10000.0) / channels))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入序列 [batch, time_steps, channels, height, width]
            
        Returns:
            添加位置编码的序列
        """
        batch_size, time_steps, channels, height, width = x.size()
        
        # 获取位置编码 [time_steps, channels]
        pos_encoding = self.pe[:time_steps, :]
        
        # 重塑为 [1, time_steps, channels, 1, 1] 以便广播
        pos_encoding = pos_encoding.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        return x + pos_encoding

class BidirectionalConvLSTM(nn.Module):
    """
    双向卷积LSTM - 同时处理前向和后向时序信息
    """
    
    def __init__(self, 
                 input_channels: int,
                 hidden_channels: int,
                 kernel_size: int = 3,
                 bias: bool = True):
        """
        初始化双向ConvLSTM
        
        Args:
            input_channels: 输入通道数
            hidden_channels: 隐藏通道数
            kernel_size: 卷积核大小
            bias: 是否使用偏置
        """
        super(BidirectionalConvLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # 前向LSTM
        self.forward_lstm = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=[hidden_channels],
            kernel_sizes=[kernel_size],
            num_layers=1,
            bias=bias
        )
        
        # 后向LSTM
        self.backward_lstm = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=[hidden_channels],
            kernel_sizes=[kernel_size],
            num_layers=1,
            bias=bias
        )
        
        # 输出通道数为双倍
        self.out_channels = hidden_channels * 2
        
        print(f"↔️  双向ConvLSTM: {input_channels} → {hidden_channels}×2 = {self.out_channels}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch, time_steps, channels, height, width]
            
        Returns:
            双向特征 [batch, time_steps, hidden_channels*2, height, width]
        """
        # 前向处理
        forward_out, _ = self.forward_lstm(x)
        forward_features = forward_out[0]  # [batch, time_steps, hidden_channels, height, width]
        
        # 后向处理 (翻转时间维度)
        x_reversed = torch.flip(x, dims=[1])
        backward_out, _ = self.backward_lstm(x_reversed)
        backward_features = backward_out[0]
        
        # 再次翻转以对齐时间
        backward_features = torch.flip(backward_features, dims=[1])
        
        # 拼接前向和后向特征
        bidirectional_features = torch.cat([forward_features, backward_features], dim=2)
        
        return bidirectional_features

class TemporalEncoder(nn.Module):
    """
    时间编码器 - 整合时间特征提取组件
    """
    
    def __init__(self,
                 input_channels: int = 256,
                 lstm_hidden_channels: int = 128,
                 use_bidirectional: bool = True,
                 use_attention: bool = True,
                 dropout_rate: float = 0.1):
        """
        初始化时间编码器
        
        Args:
            input_channels: 输入通道数 (来自空间CNN)
            lstm_hidden_channels: LSTM隐藏通道数
            use_bidirectional: 是否使用双向LSTM
            use_attention: 是否使用时间注意力
            dropout_rate: Dropout比例
        """
        super(TemporalEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.lstm_hidden_channels = lstm_hidden_channels
        self.use_bidirectional = use_bidirectional
        self.use_attention = use_attention
        
        # 输入降维 (减少计算量)
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, lstm_hidden_channels, kernel_size=1),
            nn.BatchNorm2d(lstm_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # 时序建模层
        if use_bidirectional:
            self.temporal_layer = BidirectionalConvLSTM(
                input_channels=lstm_hidden_channels,
                hidden_channels=lstm_hidden_channels//2,  # 双向后会翻倍
                kernel_size=3
            )
            temporal_out_channels = lstm_hidden_channels
        else:
            self.temporal_layer = ConvLSTM(
                input_channels=lstm_hidden_channels,
                hidden_channels=[lstm_hidden_channels],
                kernel_sizes=[3],
                num_layers=1
            )
            temporal_out_channels = lstm_hidden_channels
        
        # 时间注意力
        if use_attention:
            self.temporal_attention = TemporalAttention(
                input_channels=temporal_out_channels,
                attention_dim=64
            )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Conv2d(temporal_out_channels, input_channels, kernel_size=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out_channels = input_channels
        
        print(f"🕐 时间编码器初始化:")
        print(f"  输入通道: {input_channels}")
        print(f"  LSTM隐藏通道: {lstm_hidden_channels}")
        print(f"  双向LSTM: {use_bidirectional}")
        print(f"  时间注意力: {use_attention}")
        print(f"  输出通道: {self.out_channels}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 空间特征序列 [batch, time_steps, height, width, channels]
            
        Returns:
            (时序特征, 注意力权重)
        """
        batch_size, time_steps, height, width, channels = x.size()
        
        # 输入投影 - 逐帧处理
        # x shape: [batch, time_steps, height, width, channels]
        # 需要转换为 [batch, time_steps, channels, height, width]
        x = x.permute(0, 1, 4, 2, 3)  # [batch, time_steps, channels, height, width]
        
        projected_frames = []
        for t in range(time_steps):
            frame = self.input_projection(x[:, t, :, :, :])  # [batch, channels, height, width]
            projected_frames.append(frame)
        
        projected_sequence = torch.stack(projected_frames, dim=1)
        # projected_sequence: [batch, time_steps, lstm_hidden_channels, height, width]
        
        # 时序建模
        if self.use_bidirectional:
            temporal_features = self.temporal_layer(projected_sequence)
        else:
            temporal_out, _ = self.temporal_layer(projected_sequence)
            temporal_features = temporal_out[0]
        
        # 时间注意力
        attention_weights = None
        if self.use_attention:
            temporal_features, attention_weights = self.temporal_attention(temporal_features)
        
        # 输出投影 - 逐帧处理
        output_frames = []
        for t in range(time_steps):
            frame = self.output_projection(temporal_features[:, t, :, :, :])
            output_frames.append(frame)
        
        output_sequence = torch.stack(output_frames, dim=1)
        
        # 转换回原始格式 [batch, time_steps, height, width, channels]
        output_sequence = output_sequence.permute(0, 1, 3, 4, 2)
        
        return output_sequence, attention_weights

def test_temporal_modules():
    """测试时间序列特征提取模块"""
    print("🧪 测试时间序列特征提取模块...")
    
    # 模拟输入数据 [batch, time_steps, channels, height, width]
    batch_size = 2
    time_steps = 10  # 减少时间步以加快测试
    channels = 64
    height, width = 22, 24
    
    input_tensor = torch.randn(batch_size, time_steps, channels, height, width)
    print(f"📊 输入数据形状: {input_tensor.shape}")
    
    # 1. 测试ConvLSTM单元
    print("\n1. 测试ConvLSTM单元:")
    conv_lstm_cell = ConvLSTMCell(input_channels=channels, hidden_channels=32, kernel_size=3)
    
    # 初始化隐藏状态
    device = input_tensor.device
    h_0, c_0 = conv_lstm_cell.init_hidden(batch_size, height, width, device)
    
    # 测试单步
    h_1, c_1 = conv_lstm_cell(input_tensor[:, 0, :, :, :], (h_0, c_0))
    print(f"  隐藏状态: {h_0.shape} → {h_1.shape}")
    print(f"  细胞状态: {c_0.shape} → {c_1.shape}")
    
    # 2. 测试完整ConvLSTM
    print("\n2. 测试ConvLSTM:")
    conv_lstm = ConvLSTM(
        input_channels=channels,
        hidden_channels=[32, 64],
        kernel_sizes=[3, 3],
        num_layers=2
    )
    
    lstm_out, lstm_states = conv_lstm(input_tensor)
    print(f"  输入: {input_tensor.shape}")
    print(f"  输出: {lstm_out[0].shape}")
    print(f"  层数: {len(lstm_out)}")
    
    # 3. 测试位置编码
    print("\n3. 测试位置编码:")
    pos_encoding = PositionalEncoding(channels=channels)
    encoded_input = pos_encoding(input_tensor)
    print(f"  编码前: {input_tensor.shape}")
    print(f"  编码后: {encoded_input.shape}")
    print(f"  数值变化: {torch.mean(torch.abs(encoded_input - input_tensor)):.6f}")
    
    # 4. 测试时间注意力
    print("\n4. 测试时间注意力:")
    temporal_attention = TemporalAttention(input_channels=channels, attention_dim=32)
    attended_seq, attention_weights = temporal_attention(input_tensor)
    print(f"  输入: {input_tensor.shape}")
    print(f"  加权序列: {attended_seq.shape}")
    print(f"  注意力权重: {attention_weights.shape}")
    print(f"  权重总和: {attention_weights.sum(dim=1).mean():.3f} (应接近1.0)")
    
    # 5. 测试双向ConvLSTM
    print("\n5. 测试双向ConvLSTM:")
    bi_conv_lstm = BidirectionalConvLSTM(
        input_channels=channels,
        hidden_channels=32,
        kernel_size=3
    )
    
    bi_output = bi_conv_lstm(input_tensor)
    print(f"  输入: {input_tensor.shape}")
    print(f"  双向输出: {bi_output.shape}")
    print(f"  通道扩展: {channels} → {bi_conv_lstm.out_channels}")
    
    # 6. 测试完整时间编码器
    print("\n6. 测试完整时间编码器:")
    
    # 模拟来自空间CNN的特征 (更大的通道数)
    # 注意：这里需要使用正确的格式 [batch, time_steps, height, width, channels]
    spatial_features = torch.randn(batch_size, time_steps, height, width, 256)
    
    temporal_encoder = TemporalEncoder(
        input_channels=256,
        lstm_hidden_channels=128,
        use_bidirectional=True,
        use_attention=True,
        dropout_rate=0.1
    )
    
    # 计算参数数量
    total_params = sum(p.numel() for p in temporal_encoder.parameters())
    print(f"  模型参数数: {total_params:,}")
    
    # 前向传播
    with torch.no_grad():
        temporal_output, attention_weights = temporal_encoder(spatial_features)
        
    print(f"  输入: {spatial_features.shape}")
    print(f"  输出: {temporal_output.shape}")
    if attention_weights is not None:
        print(f"  注意力: {attention_weights.shape}")
    
    # 7. 内存和性能分析
    print("\n7. 性能分析:")
    input_memory = spatial_features.numel() * 4 / (1024**2)  # MB
    output_memory = temporal_output.numel() * 4 / (1024**2)  # MB
    print(f"  输入内存: {input_memory:.2f} MB")
    print(f"  输出内存: {output_memory:.2f} MB")
    print(f"  参数内存: {total_params * 4 / (1024**2):.2f} MB")
    
    print("\n✅ 时间序列特征提取模块测试完成!")
    return temporal_encoder, temporal_output, attention_weights

if __name__ == "__main__":
    test_temporal_modules()