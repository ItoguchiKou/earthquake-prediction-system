"""
日本地震数据网格系统 - 不规则网格版本
基于实际陆地分布的28个有效网格
"""

import numpy as np
from typing import Tuple, List, Dict, Optional

class JapanGridSystem:
    """日本地震数据不规则网格系统"""
    
    def __init__(self):
        """初始化网格系统 - 使用28个有效陆地网格"""
        # 定义有效网格（基于您提供的数据）
        self.valid_grids = {
            # 冲绳地区
            '0_4': {'lat': [24, 27], 'lon': [126, 129], 'prefectures': ['沖縄県']},
            # 九州南部
            '1_2': {'lat': [27, 30], 'lon': [129, 132], 'prefectures': ['鹿児島県']},
            '1_3': {'lat': [27, 30], 'lon': [132, 135], 'prefectures': ['鹿児島県', '宮崎県']},
            # 九州中部
            '2_2': {'lat': [30, 33], 'lon': [129, 132], 'prefectures': ['長崎県', '佐賀県', '熊本県', '鹿児島県']},
            '2_3': {'lat': [30, 33], 'lon': [132, 135], 'prefectures': ['大分県', '宮崎県', '熊本県']},
            # 九州北部・中国地方西部
            '3_2': {'lat': [33, 36], 'lon': [129, 132], 'prefectures': ['福岡県', '佐賀県', '山口県']},
            '3_3': {'lat': [33, 36], 'lon': [132, 135], 'prefectures': ['山口県', '広島県', '愛媛県', '大分県']},
            '3_4': {'lat': [33, 36], 'lon': [135, 138], 'prefectures': ['岡山県', '香川県', '徳島県', '高知県']},
            '3_5': {'lat': [33, 36], 'lon': [138, 141], 'prefectures': ['和歌山県', '三重県', '愛知県', '静岡県']},
            # 中国・近畿地方北部
            '4_2': {'lat': [36, 39], 'lon': [129, 132], 'prefectures': ['島根県']},
            '4_3': {'lat': [36, 39], 'lon': [132, 135], 'prefectures': ['島根県', '鳥取県', '広島県']},
            '4_4': {'lat': [36, 39], 'lon': [135, 138], 'prefectures': ['兵庫県', '大阪府', '京都府', '奈良県']},
            '4_5': {'lat': [36, 39], 'lon': [138, 141], 'prefectures': ['東京都', '神奈川県', '千葉県', '埼玉県', '山梨県']},
            # 中部地方
            '5_3': {'lat': [36, 39], 'lon': [132, 135], 'prefectures': ['石川県', '福井県']},
            '5_4': {'lat': [36, 39], 'lon': [135, 138], 'prefectures': ['福井県', '岐阜県', '滋賀県', '富山県']},
            '5_5': {'lat': [36, 39], 'lon': [138, 141], 'prefectures': ['長野県', '群馬県', '栃木県', '埼玉県']},
            '5_6': {'lat': [36, 39], 'lon': [141, 144], 'prefectures': ['茨城県', '千葉県', '福島県']},
            # 北陸・東北南部
            '6_4': {'lat': [36, 39], 'lon': [135, 138], 'prefectures': ['新潟県', '富山県']},
            '6_5': {'lat': [36, 39], 'lon': [138, 141], 'prefectures': ['新潟県', '福島県', '群馬県']},
            '6_6': {'lat': [36, 39], 'lon': [141, 144], 'prefectures': ['福島県', '宮城県']},
            # 東北中部
            '7_4': {'lat': [39, 42], 'lon': [135, 138], 'prefectures': ['秋田県', '山形県']},
            '7_5': {'lat': [39, 42], 'lon': [138, 141], 'prefectures': ['山形県', '秋田県']},
            '7_6': {'lat': [39, 42], 'lon': [141, 144], 'prefectures': ['宮城県', '岩手県']},
            # 東北北部
            '8_5': {'lat': [39, 42], 'lon': [138, 141], 'prefectures': ['青森県', '秋田県']},
            '8_6': {'lat': [39, 42], 'lon': [141, 144], 'prefectures': ['岩手県', '青森県']},
            # 北海道
            '9_5': {'lat': [42, 45], 'lon': [138, 141], 'prefectures': ['北海道（道西・道央）']},
            '9_6': {'lat': [42, 45], 'lon': [141, 144], 'prefectures': ['北海道（道央・道南）']},
            '9_7': {'lat': [42, 45], 'lon': [144, 147], 'prefectures': ['北海道（道東）']},
        }
        
        # 创建网格ID到索引的映射
        self.grid_id_to_index = {grid_id: idx for idx, grid_id in enumerate(sorted(self.valid_grids.keys()))}
        self.index_to_grid_id = {idx: grid_id for grid_id, idx in self.grid_id_to_index.items()}
        
        # 网格数量
        self.num_valid_grids = len(self.valid_grids)
        
        # 边界范围（用于快速过滤）
        self.min_lat = 24.0
        self.max_lat = 45.0
        self.min_lon = 126.0
        self.max_lon = 147.0
        
        # 网格分辨率
        self.grid_resolution = 3.0
        
        # 为了兼容性，保留这些属性（但值改为有效网格的维度）
        self.lat_grids = 10  # 0-9的索引范围
        self.lon_grids = 8   # 0-7的索引范围
        self.total_grids = self.num_valid_grids  # 实际有效网格数
        
        # 创建快速查找表
        self._create_lookup_tables()
        
        print(f"不规则网格系统初始化完成:")
        print(f"  有效网格数: {self.num_valid_grids}")
        print(f"  纬度范围: {self.min_lat}° - {self.max_lat}°")
        print(f"  经度范围: {self.min_lon}° - {self.max_lon}°")
        print(f"  网格分辨率: {self.grid_resolution}°×{self.grid_resolution}°")
    
    def _create_lookup_tables(self):
        """创建坐标到网格的快速查找表"""
        # 创建规则网格索引到有效网格的映射
        self.regular_to_valid = {}
        
        for grid_id, grid_info in self.valid_grids.items():
            lat_idx, lon_idx = map(int, grid_id.split('_'))
            self.regular_to_valid[(lat_idx, lon_idx)] = grid_id
    
    def coords_to_grid(self, latitude: float, longitude: float) -> Tuple[int, int]:
        """
        将经纬度坐标转换为网格索引
        
        Args:
            latitude: 纬度
            longitude: 经度
            
        Returns:
            (lat_idx, lon_idx): 网格索引，(-1, -1)表示超出范围或无效网格
        """
        # 检查是否在有效范围内
        if not (self.min_lat <= latitude <= self.max_lat and 
                self.min_lon <= longitude <= self.max_lon):
            return (-1, -1)
        
        # 计算规则网格索引
        lat_idx = int((latitude - 24.0) / self.grid_resolution)
        lon_idx = int((longitude - 123.0) / self.grid_resolution)
        
        # 检查是否是有效网格
        grid_key = (lat_idx, lon_idx)
        if grid_key in self.regular_to_valid:
            return grid_key
        else:
            return (-1, -1)
    
    def coords_to_valid_index(self, latitude: float, longitude: float) -> int:
        """
        将经纬度坐标转换为有效网格的线性索引
        
        Args:
            latitude: 纬度
            longitude: 经度
            
        Returns:
            有效网格索引 (0-27)，-1表示无效
        """
        lat_idx, lon_idx = self.coords_to_grid(latitude, longitude)
        if lat_idx == -1:
            return -1
        
        grid_id = f"{lat_idx}_{lon_idx}"
        return self.grid_id_to_index.get(grid_id, -1)
    
    def grid_to_coords(self, lat_idx: int, lon_idx: int) -> Tuple[float, float]:
        """
        将网格索引转换为网格中心点经纬度
        
        Args:
            lat_idx: 纬度网格索引
            lon_idx: 经度网格索引
            
        Returns:
            (center_lat, center_lon): 网格中心点坐标
        """
        grid_id = f"{lat_idx}_{lon_idx}"
        if grid_id in self.valid_grids:
            grid_info = self.valid_grids[grid_id]
            center_lat = (grid_info['lat'][0] + grid_info['lat'][1]) / 2
            center_lon = (grid_info['lon'][0] + grid_info['lon'][1]) / 2
            return (center_lat, center_lon)
        return (None, None)
    
    def get_grid_bounds(self, lat_idx: int, lon_idx: int) -> Tuple[float, float, float, float]:
        """
        获取网格边界
        
        Args:
            lat_idx: 纬度网格索引
            lon_idx: 经度网格索引
            
        Returns:
            (min_lat, max_lat, min_lon, max_lon): 网格边界
        """
        grid_id = f"{lat_idx}_{lon_idx}"
        if grid_id in self.valid_grids:
            grid_info = self.valid_grids[grid_id]
            return (grid_info['lat'][0], grid_info['lat'][1], 
                   grid_info['lon'][0], grid_info['lon'][1])
        return (None, None, None, None)
    
    def get_neighbor_grids(self, lat_idx: int, lon_idx: int, radius: int = 1) -> List[Tuple[int, int]]:
        """
        获取指定网格的邻域网格（只返回有效网格）
        
        Args:
            lat_idx: 纬度网格索引
            lon_idx: 经度网格索引  
            radius: 邻域半径（网格数）
            
        Returns:
            邻域网格索引列表
        """
        neighbors = []
        
        for dlat in range(-radius, radius + 1):
            for dlon in range(-radius, radius + 1):
                new_lat_idx = lat_idx + dlat
                new_lon_idx = lon_idx + dlon
                
                # 检查是否是有效网格
                grid_id = f"{new_lat_idx}_{new_lon_idx}"
                if grid_id in self.valid_grids:
                    neighbors.append((new_lat_idx, new_lon_idx))
        
        return neighbors
    
    def get_all_valid_grids(self) -> List[Tuple[int, int]]:
        """获取所有有效网格的索引"""
        valid_indices = []
        for grid_id in self.valid_grids:
            lat_idx, lon_idx = map(int, grid_id.split('_'))
            valid_indices.append((lat_idx, lon_idx))
        return sorted(valid_indices)
    
    def get_grid_info(self) -> dict:
        """
        获取网格系统信息
        
        Returns:
            网格系统信息字典
        """
        return {
            'lat_range': (self.min_lat, self.max_lat),
            'lon_range': (self.min_lon, self.max_lon),
            'grid_resolution': self.grid_resolution,
            'num_valid_grids': self.num_valid_grids,
            'valid_grid_ids': list(self.valid_grids.keys()),
            'index_mapping': self.grid_id_to_index
        }
    
    def create_dense_grid_matrix(self, sparse_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        将稀疏的有效网格数据转换为密集矩阵（用于CNN）
        
        Args:
            sparse_data: {grid_id: feature_vector} 的字典
            
        Returns:
            密集矩阵 [10, 8, features]，无效网格填充0
        """
        if not sparse_data:
            return np.zeros((10, 8, 1))
        
        # 获取特征维度
        sample_features = next(iter(sparse_data.values()))
        feature_dim = sample_features.shape[-1] if len(sample_features.shape) > 0 else 1
        
        # 创建密集矩阵
        dense_matrix = np.zeros((10, 8, feature_dim))
        
        # 填充有效网格数据
        for grid_id, features in sparse_data.items():
            if grid_id in self.valid_grids:
                lat_idx, lon_idx = map(int, grid_id.split('_'))
                dense_matrix[lat_idx, lon_idx] = features
        
        return dense_matrix
    
    def extract_valid_grids_from_matrix(self, dense_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        从密集矩阵中提取有效网格数据
        
        Args:
            dense_matrix: 密集矩阵 [10, 8, features]
            
        Returns:
            {grid_id: feature_vector} 的字典
        """
        sparse_data = {}
        
        for grid_id in self.valid_grids:
            lat_idx, lon_idx = map(int, grid_id.split('_'))
            if lat_idx < dense_matrix.shape[0] and lon_idx < dense_matrix.shape[1]:
                sparse_data[grid_id] = dense_matrix[lat_idx, lon_idx]
        
        return sparse_data