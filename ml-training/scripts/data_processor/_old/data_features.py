"""
地震数据特征工程模块
为Random Forest模型准备训练特征
"""

import pandas as pd
import numpy as np
from datetime import timedelta

def create_earthquake_features(df):
    """
    为地震数据创建Random Forest训练特征
    
    Args:
        df: 地震数据DataFrame
        
    Returns:
        特征DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    # 确保时间列是datetime类型
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    features_list = []
    
    # 为每个地震事件创建特征
    for i in range(len(df)):
        current_earthquake = df.iloc[i]
        current_time = current_earthquake['time']
        
        # 历史时间窗口
        window_30d = current_time - timedelta(days=30)
        window_7d = current_time - timedelta(days=7)
        window_1d = current_time - timedelta(days=1)
        
        # 当前时间之前的数据
        past_data = df[df['time'] < current_time]
        
        # 基础特征
        features = {
            # 地理特征
            'latitude': current_earthquake['latitude'],
            'longitude': current_earthquake['longitude'],
            'depth': current_earthquake['depth'],
            
            # 时间特征
            'year': current_time.year,
            'month': current_time.month,
            'day_of_year': current_time.dayofyear,
            'hour': current_time.hour,
            'day_of_week': current_time.dayofweek,
            
            # 历史地震频率特征（30天内）
            'earthquakes_30d': len(past_data[past_data['time'] >= window_30d]),
            'earthquakes_7d': len(past_data[past_data['time'] >= window_7d]),
            'earthquakes_1d': len(past_data[past_data['time'] >= window_1d]),
        }
        
        # 历史震级统计特征
        if len(past_data) > 0:
            past_30d = past_data[past_data['time'] >= window_30d]
            past_7d = past_data[past_data['time'] >= window_7d]
            
            # 30天内震级统计
            if len(past_30d) > 0:
                features.update({
                    'avg_magnitude_30d': past_30d['magnitude'].mean(),
                    'max_magnitude_30d': past_30d['magnitude'].max(),
                    'std_magnitude_30d': past_30d['magnitude'].std(),
                })
            else:
                features.update({
                    'avg_magnitude_30d': 0,
                    'max_magnitude_30d': 0,
                    'std_magnitude_30d': 0,
                })
            
            # 7天内震级统计
            if len(past_7d) > 0:
                features.update({
                    'avg_magnitude_7d': past_7d['magnitude'].mean(),
                    'max_magnitude_7d': past_7d['magnitude'].max(),
                })
            else:
                features.update({
                    'avg_magnitude_7d': 0,
                    'max_magnitude_7d': 0,
                })
            
            # 最近地震时间间隔
            features['days_since_last_earthquake'] = (current_time - past_data['time'].max()).days
            
            # 附近区域地震特征（0.5度范围内）
            nearby_earthquakes = past_data[
                (abs(past_data['latitude'] - current_earthquake['latitude']) <= 0.5) &
                (abs(past_data['longitude'] - current_earthquake['longitude']) <= 0.5) &
                (past_data['time'] >= window_30d)
            ]
            features['nearby_earthquakes_30d'] = len(nearby_earthquakes)
            
        else:
            # 如果没有历史数据，使用默认值
            features.update({
                'avg_magnitude_30d': 0,
                'max_magnitude_30d': 0,
                'std_magnitude_30d': 0,
                'avg_magnitude_7d': 0,
                'max_magnitude_7d': 0,
                'days_since_last_earthquake': 999,
                'nearby_earthquakes_30d': 0,
            })
        
        # 目标变量（当前地震的震级）
        features['target_magnitude'] = current_earthquake['magnitude']
        
        # 添加原始数据索引
        features['original_index'] = i
        features['earthquake_time'] = current_time
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def prepare_training_data(features_df, min_samples=50):
    """
    准备Random Forest训练数据
    
    Args:
        features_df: 特征DataFrame
        min_samples: 最少样本数量
        
    Returns:
        X: 特征矩阵, y: 目标变量
    """
    if len(features_df) < min_samples:
        print(f"警告: 样本数量 ({len(features_df)}) 少于最小要求 ({min_samples})")
        return None, None
    
    # 选择训练特征
    feature_columns = [
        'latitude', 'longitude', 'depth',
        'month', 'day_of_year', 'hour', 'day_of_week',
        'earthquakes_30d', 'earthquakes_7d', 'earthquakes_1d',
        'avg_magnitude_30d', 'max_magnitude_30d', 'std_magnitude_30d',
        'avg_magnitude_7d', 'max_magnitude_7d',
        'days_since_last_earthquake', 'nearby_earthquakes_30d'
    ]
    
    # 检查特征列是否存在
    available_features = [col for col in feature_columns if col in features_df.columns]
    
    X = features_df[available_features].copy()
    y = features_df['target_magnitude'].copy()
    
    # 处理缺失值
    X = X.fillna(0)
    
    # 处理无穷大值
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, y