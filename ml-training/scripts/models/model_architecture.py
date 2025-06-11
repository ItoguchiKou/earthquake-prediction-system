"""
地震预测主模型架构
整合所有子模块构建完整的CNN+LSTM+Attention+多任务学习模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 导入所有子模块
from .model_utils import init_weights
from .loss_functions import EarthquakeLoss
from .spatial_modules import MultiScaleCNN
from .temporal_modules import TemporalEncoder
from .attention_modules import EarthquakeAttentionFusion
from .multi_task_heads import MultiTaskOutputLayer, PredictionPostProcessor

class EarthquakePredictionModel(nn.Module):
    """
    地震预测主模型
    CNN + ConvLSTM + Attention + 多任务学习的完整架构
    """
    
    def __init__(self,
                 # 输入参数
                 input_channels: int = 8,
                 input_time_steps: int = 90,
                 input_height: int = 22,
                 input_width: int = 24,
                 
                 # 空间CNN参数
                 spatial_base_channels: int = 64,
                 spatial_num_scales: int = 3,
                 
                 # 时间编码器参数
                 temporal_hidden_channels: int = 128,
                 use_bidirectional_lstm: bool = True,
                 use_temporal_attention: bool = True,
                 
                 # 注意力融合参数
                 attention_fusion_dim: int = 512,
                 attention_num_heads: int = 8,
                 
                 # 多任务输出参数
                 shared_feature_dim: int = 128,
                 task_hidden_dim: int = 64,
                 prediction_windows: List[int] = [7, 14, 30],
                 magnitude_ranges: List[Tuple[float, float]] = [(3.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 10.0)],
                 
                 # 训练参数
                 dropout_rate: float = 0.2,
                 use_post_processing: bool = True):
        """
        初始化地震预测模型
        
        Args:
            input_channels: 输入特征通道数 (8个地震特征)
            input_time_steps: 输入时间步长 (90天历史)
            input_height: 输入网格高度 (22个纬度网格)
            input_width: 输入网格宽度 (24个经度网格)
            spatial_base_channels: 空间CNN基础通道数
            spatial_num_scales: 空间多尺度数量
            temporal_hidden_channels: 时间编码器隐藏通道数
            use_bidirectional_lstm: 是否使用双向LSTM
            use_temporal_attention: 是否使用时间注意力
            attention_fusion_dim: 注意力融合后维度
            attention_num_heads: 多头注意力头数
            shared_feature_dim: 多任务共享特征维度
            task_hidden_dim: 任务特定隐藏层维度
            prediction_windows: 预测时间窗口列表
            magnitude_ranges: 震级范围列表
            dropout_rate: Dropout比例
            use_post_processing: 是否使用预测后处理
        """
        super(EarthquakePredictionModel, self).__init__()
        
        # 保存配置参数
        self.config = {
            'input_channels': input_channels,
            'input_time_steps': input_time_steps,
            'input_height': input_height,
            'input_width': input_width,
            'spatial_base_channels': spatial_base_channels,
            'spatial_num_scales': spatial_num_scales,
            'temporal_hidden_channels': temporal_hidden_channels,
            'use_bidirectional_lstm': use_bidirectional_lstm,
            'use_temporal_attention': use_temporal_attention,
            'attention_fusion_dim': attention_fusion_dim,
            'attention_num_heads': attention_num_heads,
            'shared_feature_dim': shared_feature_dim,
            'task_hidden_dim': task_hidden_dim,
            'prediction_windows': prediction_windows,
            'magnitude_ranges': magnitude_ranges,
            'dropout_rate': dropout_rate,
            'use_post_processing': use_post_processing
        }
        
        # 计算中间维度
        self.spatial_output_channels = spatial_base_channels * (2 ** (spatial_num_scales - 1))
        self.temporal_output_channels = self.spatial_output_channels  # TemporalEncoder保持维度
        
        # 1. 空间特征提取模块 (CNN)
        self.spatial_encoder = MultiScaleCNN(
            in_channels=input_channels,
            base_channels=spatial_base_channels,
            num_scales=spatial_num_scales
        )
        
        # 2. 时间特征提取模块 (ConvLSTM + Attention)
        self.temporal_encoder = TemporalEncoder(
            input_channels=self.spatial_output_channels,
            lstm_hidden_channels=temporal_hidden_channels,
            use_bidirectional=use_bidirectional_lstm,
            use_attention=use_temporal_attention,
            dropout_rate=dropout_rate
        )
        
        # 3. 注意力融合模块
        self.attention_fusion = EarthquakeAttentionFusion(
            spatial_channels=self.spatial_output_channels,
            temporal_channels=self.temporal_output_channels,
            fusion_dim=attention_fusion_dim,
            num_heads=attention_num_heads
        )
        
        # 4. 多任务输出模块
        self.multi_task_head = MultiTaskOutputLayer(
            input_channels=attention_fusion_dim,
            shared_feature_dim=shared_feature_dim,
            task_hidden_dim=task_hidden_dim,
            prediction_windows=prediction_windows,
            magnitude_ranges=magnitude_ranges
        )
        
        # 5. 预测后处理模块 (可选)
        if use_post_processing:
            self.post_processor = PredictionPostProcessor(
                num_tasks=len(prediction_windows) * len(magnitude_ranges)
            )
        else:
            self.post_processor = None
        
        # 6. 损失函数
        self.loss_function = EarthquakeLoss(
            focal_weight=1.0,
            consistency_weight=0.1,
            uncertainty_weighting=True
        )
        
        # === 3x3网格优化：添加额外的正则化 ===
        # 检测是否使用3x3网格（输入高度和宽度较小）
        self.is_coarse_grid = (input_height <= 10 and input_width <= 10)
        
        if self.is_coarse_grid:
            print("🔧 检测到3x3网格配置，启用额外正则化")
            
            # 为粗网格添加额外的dropout层
            self.extra_spatial_dropout = nn.Dropout2d(0.5)
            self.extra_temporal_dropout = nn.Dropout(0.5)
            
            # 调整dropout率
            enhanced_dropout_rate = min(dropout_rate * 2.0, 0.6)
            
            # 为关键层添加噪声注入（训练时的正则化）
            self.training_noise_std = 0.05
            
            # 定义L2正则化强度和特征范数约束
            self.l2_regularization_strength = 0.001
            self.feature_norm_constraint = 10.0
            
            print(f"  - 额外空间Dropout: 0.5")
            print(f"  - 额外时间Dropout: 0.5")
            print(f"  - 增强Dropout率: {enhanced_dropout_rate}")
            print(f"  - 训练噪声标准差: {self.training_noise_std}")
            print(f"  - L2正则化强度: {self.l2_regularization_strength}")
            print(f"  - 特征范数约束: {self.feature_norm_constraint}")

            # 添加谱归一化到关键层（稳定训练）
            self._apply_spectral_norm()
        else:
            self.extra_spatial_dropout = None
            self.extra_temporal_dropout = None
            self.training_noise_std = 0.0
            self.l2_regularization_strength = 0.0
            self.feature_norm_constraint = float('inf')
        
        # 初始化权重
        self.apply(lambda m: init_weights(m, 'kaiming_uniform'))
        
        self.float()  # 确保所有参数为float32
        torch.set_default_tensor_type(torch.FloatTensor)  # 设置默认张量类型
        
        # 打印模型信息
        self._print_model_info()
    
    # 应用谱归一化
    def _apply_spectral_norm(self):
        """对关键层应用谱归一化"""
        # 对空间编码器的卷积层应用谱归一化
        for module in self.spatial_encoder.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.utils.spectral_norm(module)
        
        # 对注意力融合的关键层应用谱归一化
        for module in self.attention_fusion.modules():
            if isinstance(module, nn.Linear) and module.out_features > 256:
                torch.nn.utils.spectral_norm(module)


    def _print_model_info(self):
        """打印模型架构信息"""
        print("\n" + "="*80)
        print("🌍 地震预测模型架构信息")
        print("="*80)
        
        print(f"📊 输入配置:")
        print(f"  输入维度: [{self.config['input_time_steps']}, {self.config['input_height']}, {self.config['input_width']}, {self.config['input_channels']}]")
        print(f"  网格分辨率: 1°×1° (共{self.config['input_height']}×{self.config['input_width']}={self.config['input_height']*self.config['input_width']}个网格)")
        print(f"  时间跨度: {self.config['input_time_steps']}天历史数据")
        
        print(f"\n🧠 网络架构:")
        print(f"  空间编码器: MultiScaleCNN ({self.config['input_channels']} → {self.spatial_output_channels}通道)")
        print(f"  时间编码器: TemporalEncoder (双向LSTM: {self.config['use_bidirectional_lstm']}, 注意力: {self.config['use_temporal_attention']})")
        print(f"  注意力融合: EarthquakeAttentionFusion ({self.config['attention_fusion_dim']}维)")
        print(f"  多任务输出: {len(self.config['prediction_windows'])}×{len(self.config['magnitude_ranges'])}={len(self.config['prediction_windows'])*len(self.config['magnitude_ranges'])}个任务")
        
        print(f"\n🎯 预测任务:")
        print(f"  时间窗口: {self.config['prediction_windows']} 天")
        print(f"  震级范围: {self.config['magnitude_ranges']}")
        
        # 计算总参数数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📈 模型规模:")
        print(f"  总参数数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型大小: {total_params * 4 / (1024**2):.2f} MB")
        
        print("="*80)
    
    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播 - 添加了3x3网格的额外正则化
        
        Args:
            x: 输入数据 [batch, time_steps, height, width, channels]
            return_intermediates: 是否返回中间结果
            
        Returns:
            预测结果或包含中间结果的字典
        """
        batch_size, time_steps, height, width, channels = x.size()
        
        # 验证输入维度
        if input_height != 10 or input_width != 8:
            print(f"⚠️ 警告：输入维度({input_height}×{input_width})不是标准的10×8不规则网格")
            print("   模型可能无法正确处理数据！")
        
        # === 3x3网格优化：训练时添加输入噪声 ===
        if self.training and self.is_coarse_grid and self.training_noise_std > 0:
            # 使用更复杂的噪声模式
            noise = torch.randn_like(x) * self.training_noise_std
            # 添加空间相关噪声
            spatial_noise = F.interpolate(
                torch.randn(x.size(0), x.size(1), 4, 4, x.size(4)).to(x.device),
                size=(x.size(2), x.size(3)),
                mode='bilinear',
                align_corners=False
            )
            x = x + noise + spatial_noise * 0.02
  
        
        # 1. 空间特征提取
        spatial_features = self.spatial_encoder(x)
        # 输出: [batch, time_steps, height, width, spatial_output_channels]
        
        # === 3x3网格优化：应用额外的空间dropout ===
        if self.training and self.is_coarse_grid and self.extra_spatial_dropout is not None:
            # 对每个时间步应用空间dropout
            spatial_features_list = []
            for t in range(time_steps):
                feat = spatial_features[:, t, :, :, :].permute(0, 3, 1, 2)  # [B, C, H, W]
                feat = self.extra_spatial_dropout(feat)
                feat = feat.permute(0, 2, 3, 1)  # [B, H, W, C]
                spatial_features_list.append(feat)
            spatial_features = torch.stack(spatial_features_list, dim=1)
        
        # 2. 时间特征提取
        temporal_features, temporal_attention = self.temporal_encoder(spatial_features)
        # 输出: [batch, time_steps, height, width, temporal_output_channels]
        
        # === 3x3网格优化：应用额外的时间dropout ===
        if self.training and self.is_coarse_grid and self.extra_temporal_dropout is not None:
            # 对时间维度应用dropout
            b, t, h, w, c = temporal_features.shape
            temporal_features = temporal_features.view(b, t, -1)
            temporal_features = self.extra_temporal_dropout(temporal_features)
            temporal_features = temporal_features.view(b, t, h, w, c)
        
        # === 添加特征范数约束 ===
        if self.training and self.is_coarse_grid:
            # 对中间特征应用范数约束
            with torch.no_grad():
                spatial_norm = torch.norm(spatial_features, p=2, dim=(2,3,4), keepdim=True)
                spatial_features = spatial_features / torch.clamp(spatial_norm / self.feature_norm_constraint, min=1.0)
                
                temporal_norm = torch.norm(temporal_features, p=2, dim=(2,3,4), keepdim=True)
                temporal_features = temporal_features / torch.clamp(temporal_norm / self.feature_norm_constraint, min=1.0)

        # 转换为注意力融合需要的格式 [batch, time_steps, channels, height, width]
        spatial_features_for_attention = spatial_features.permute(0, 1, 4, 2, 3)
        temporal_features_for_attention = temporal_features.permute(0, 1, 4, 2, 3)
        

        # 3. 注意力融合
        fused_features, attention_weights = self.attention_fusion(
            spatial_features_for_attention, temporal_features_for_attention
        )
        # 输出: [batch, time_steps, fusion_dim, height, width]
        
        # 4. 多任务预测
        task_predictions = self.multi_task_head(fused_features)
    
        # 5. 预测后处理 (可选)
        all_predictions = task_predictions['all_tasks']  # [batch, num_tasks]
        
        if self.post_processor is not None:
            # 确保传递的是连续内存的张量
            final_predictions = self.post_processor(all_predictions.contiguous())
        else:
            final_predictions = all_predictions
        
        # 确保返回的张量是连续的且独立的
        final_predictions = final_predictions.contiguous()
        
        if return_intermediates:
            return {
                'predictions': final_predictions,
                'raw_predictions': all_predictions,
                'task_predictions': task_predictions,
                'spatial_features': spatial_features,
                'temporal_features': temporal_features,
                'fused_features': fused_features,
                'temporal_attention': temporal_attention,
                'attention_weights': attention_weights
            }
        else:
            return final_predictions
    
    # 计算L2正则化损失
    def get_l2_regularization_loss(self) -> torch.Tensor:
        """计算L2正则化损失"""
        if not self.is_coarse_grid:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        l2_loss = 0.0
        for name, param in self.named_parameters():
            if 'weight' in name and param.requires_grad:
                l2_loss += torch.norm(param, p=2)
        
        return l2_loss * self.l2_regularization_strength

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算损失 - 包含L2正则化
        
        Args:
            predictions: 模型预测 [batch, num_tasks]
            targets: 真实标签 [batch, num_tasks]
            
        Returns:
            损失信息字典
        """
        # 获取基础损失
        loss_dict = self.loss_function(predictions, targets)
        
        # 添加L2正则化损失
        if self.training and self.is_coarse_grid:
            l2_loss = self.get_l2_regularization_loss()
            loss_dict['l2_loss'] = l2_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + l2_loss
        
        return loss_dict
    
    def predict_probabilities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测地震发生概率
        
        Args:
            x: 输入数据 [batch, time_steps, height, width, channels]
            
        Returns:
            预测概率字典
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
            
            # 按时间窗口分组
            time_grouped = self.multi_task_head.get_predictions_by_time_window(predictions)
            
            # 按震级分组  
            mag_grouped = self.multi_task_head.get_predictions_by_magnitude(predictions)
            
            return {
                'all_predictions': predictions,
                'by_time_window': time_grouped,
                'by_magnitude': mag_grouped,
                'task_names': self.multi_task_head.task_names
            }
    
    def predict_single_region(self, x: torch.Tensor, lat_range: Tuple[int, int], lon_range: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        预测特定区域的地震概率
        
        Args:
            x: 输入数据 [batch, time_steps, height, width, channels]
            lat_range: 纬度网格范围 (start, end)
            lon_range: 经度网格范围 (start, end)
            
        Returns:
            区域预测结果
        """
        # 提取区域数据
        lat_start, lat_end = lat_range
        lon_start, lon_end = lon_range
        
        region_data = x[:, :, lat_start:lat_end, lon_start:lon_end, :]
        
        # 如果区域太小，进行填充
        if region_data.size(2) < self.config['input_height'] or region_data.size(3) < self.config['input_width']:
            # 使用零填充扩展到标准尺寸
            padded_data = torch.zeros_like(x)
            padded_data[:, :, :region_data.size(2), :region_data.size(3), :] = region_data
            region_data = padded_data
        
        # 进行预测
        predictions = self.predict_probabilities(region_data)
        
        # 添加区域信息
        predictions['region_info'] = {
            'lat_range': lat_range,
            'lon_range': lon_range,
            'region_size': (lat_end - lat_start, lon_end - lon_start)
        }
        
        return predictions
    
    def get_model_summary(self) -> Dict[str, any]:
        """
        获取模型摘要信息
        
        Returns:
            模型摘要字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 计算各模块参数
        module_params = {}
        for name, module in self.named_children():
            module_params[name] = sum(p.numel() for p in module.parameters())
        
        return {
            'model_name': 'EarthquakePredictionModel',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024**2),
            'module_parameters': module_params,
            'config': self.config,
            'task_info': self.multi_task_head.get_task_info()
        }
    
    def save_model(self, filepath: str, save_optimizer: bool = False, optimizer: Optional[torch.optim.Optimizer] = None):
        """
        保存模型
        
        Args:
            filepath: 保存路径
            save_optimizer: 是否保存优化器状态
            optimizer: 优化器实例
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'model_summary': self.get_model_summary()
        }
        
        if save_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, filepath)
        print(f"💾 模型已保存: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: Optional[torch.device] = None):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            device: 目标设备
            
        Returns:
            加载的模型实例
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['model_config']
        
        # 创建模型实例
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"📂 模型已加载: {filepath}")
        print(f"   设备: {device}")
        print(f"   参数数: {checkpoint['model_summary']['total_parameters']:,}")
        
        return model, checkpoint.get('optimizer_state_dict', None)

def test_earthquake_prediction_model():
    """测试完整的地震预测模型"""
    print("🧪 测试完整地震预测模型...")
    
    # 模拟输入数据 [batch, time_steps, height, width, channels]
    batch_size = 2
    time_steps = 90
    height, width = 22, 24
    channels = 8
    
    input_data = torch.randn(batch_size, time_steps, height, width, channels)
    print(f"📊 输入数据形状: {input_data.shape}")
    
    # 创建模型
    print("\n🏗️  创建地震预测模型:")
    model = EarthquakePredictionModel(
        input_channels=channels,
        input_time_steps=time_steps,
        input_height=height,
        input_width=width,
        spatial_base_channels=32,  # 减少通道数以加快测试
        spatial_num_scales=2,
        temporal_hidden_channels=64,
        attention_fusion_dim=256,
        shared_feature_dim=64,
        task_hidden_dim=32
    )
    
    # 获取模型摘要
    summary = model.get_model_summary()
    print(f"\n📋 模型摘要:")
    print(f"  总参数数: {summary['total_parameters']:,}")
    print(f"  模型大小: {summary['model_size_mb']:.2f} MB")
    
    # 1. 测试前向传播
    print("\n🔄 测试前向传播:")
    model.eval()
    with torch.no_grad():
        # 基础预测
        predictions = model(input_data)
        print(f"  预测输出: {predictions.shape}")
        print(f"  预测范围: [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        # 详细预测 (包含中间结果)
        detailed_results = model(input_data, return_intermediates=True)
        print(f"  中间结果数: {len(detailed_results)}")
    
    # 2. 测试概率预测
    print("\n🎯 测试概率预测:")
    prob_results = model.predict_probabilities(input_data)
    print(f"  全任务预测: {prob_results['all_predictions'].shape}")
    print(f"  时间窗口分组: {len(prob_results['by_time_window'])} 组")
    print(f"  震级分组: {len(prob_results['by_magnitude'])} 组")
    
    # 3. 测试区域预测
    print("\n🗺️  测试区域预测:")
    region_results = model.predict_single_region(
        input_data, 
        lat_range=(5, 15), 
        lon_range=(8, 18)
    )
    print(f"  区域预测: {region_results['all_predictions'].shape}")
    print(f"  区域大小: {region_results['region_info']['region_size']}")
    
    # 4. 测试损失计算
    print("\n💔 测试损失计算:")
    # 创建模拟标签
    targets = torch.randint(0, 2, (batch_size, 12)).float()
    loss_info = model.compute_loss(predictions, targets)
    print(f"  总损失: {loss_info['total_loss']:.4f}")
    print(f"  Focal损失: {loss_info['focal_loss']:.4f}")
    print(f"  一致性损失: {loss_info['consistency_loss']:.4f}")
    
    # 5. 测试模型保存/加载
    print("\n💾 测试模型保存/加载:")
    save_path = "test_earthquake_model.pth"
    
    # 保存模型
    model.save_model(save_path)
    
    # 加载模型
    loaded_model, _ = EarthquakePredictionModel.load_model(save_path)
    
    # 验证加载的模型
    with torch.no_grad():
        original_pred = model(input_data)
        loaded_pred = loaded_model(input_data)
        difference = torch.abs(original_pred - loaded_pred).max()
        print(f"  加载验证: 最大差异 = {difference:.6f}")
    
    # 6. 性能分析
    print("\n📊 性能分析:")
    input_memory = input_data.numel() * 4 / (1024**2)
    output_memory = predictions.numel() * 4 / (1024**2)
    model_memory = summary['model_size_mb']
    
    print(f"  输入内存: {input_memory:.2f} MB")
    print(f"  输出内存: {output_memory:.2f} MB")
    print(f"  模型内存: {model_memory:.2f} MB")
    print(f"  总内存估算: {input_memory + output_memory + model_memory:.2f} MB")
    
    print("\n✅ 完整地震预测模型测试完成!")
    
    # 清理临时文件
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
    
    return model, predictions, prob_results

if __name__ == "__main__":
    test_earthquake_prediction_model()