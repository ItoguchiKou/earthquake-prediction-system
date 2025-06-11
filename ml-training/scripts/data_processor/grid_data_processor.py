"""
网格化地震数据处理主流程
加载原始CSV数据，映射到网格系统
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from typing import List
import warnings
warnings.filterwarnings('ignore')

from grid_system import JapanGridSystem

class GridDataProcessor:
    """网格化地震数据处理器"""

    def __init__(self, data_dir: str = "/path/to/your/data"):
        """
        初始化数据处理器

        Args:
            data_dir: 数据目录绝对路径
        """
        self.data_dir = os.path.abspath(data_dir)
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed_grid")

        # 创建输出目录
        os.makedirs(self.processed_dir, exist_ok=True)

        # 初始化网格系统
        self.grid_system = JapanGridSystem()

        print(f"数据处理器初始化完成:")
        print(f"  原始数据目录: {self.raw_dir}")
        print(f"  处理后数据目录: {self.processed_dir}")

    def discover_data_files(self) -> List[str]:
        """
        发现并排序原始数据文件

        Returns:
            排序后的数据文件路径列表
        """
        print(":mag: 搜索原始数据文件...")

        # 搜索5年范围文件
        pattern_5year = os.path.join(self.raw_dir, "rawData_*-*.csv")
        files_5year = glob.glob(pattern_5year)

        # 搜索单年文件 (2025)
        pattern_single = os.path.join(self.raw_dir, "rawData_2025.csv")
        files_single = glob.glob(pattern_single)

        all_files = files_5year + files_single

        if not all_files:
            print(f":x: 在 {self.raw_dir} 未找到数据文件")
            return []

        # 按年份排序
        def extract_year(filepath):
            filename = os.path.basename(filepath)
            if "2025.csv" in filename:
                return 2025
            else:
                # 提取起始年份
                year_part = filename.replace('rawData_', '').replace('.csv', '')
                start_year = int(year_part.split('-')[0])
                return start_year

        sorted_files = sorted(all_files, key=extract_year)

        print(f":file_folder: 发现 {len(sorted_files)} 个数据文件:")
        for file_path in sorted_files:
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  :white_check_mark: {filename} ({file_size:.1f} MB)")

        return sorted_files

    def load_raw_data(self, files: List[str], min_magnitude: float = 3.0) -> pd.DataFrame:
        """
        加载并合并原始地震数据

        Args:
            files: 数据文件路径列表
            min_magnitude: 最小震级过滤

        Returns:
            合并后的地震数据DataFrame
        """
        print(f"\n:open_file_folder: 加载原始地震数据 (震级 ≥ {min_magnitude})...")

        all_data = []
        total_records = 0

        for i, file_path in enumerate(files, 1):
            filename = os.path.basename(file_path)
            print(f"  处理 [{i}/{len(files)}]: {filename}")

            try:
                # 读取CSV
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                original_count = len(df)

                # 数据清洗
                df = df.dropna(subset=['latitude', 'longitude', 'magnitude', 'time'])
                df = df[df['magnitude'] >= min_magnitude]

                # 时间转换
                df['time'] = pd.to_datetime(df['time'])

                # 过滤日本范围内的地震
                df = df[
                    (df['latitude'] >= self.grid_system.min_lat) &
                    (df['latitude'] <= self.grid_system.max_lat) &
                    (df['longitude'] >= self.grid_system.min_lon) &
                    (df['longitude'] <= self.grid_system.max_lon)
                ]

                cleaned_count = len(df)

                if cleaned_count > 0:
                    all_data.append(df)
                    total_records += cleaned_count
                    print(f"    :white_check_mark: 原始: {original_count:,} → 清洗后: {cleaned_count:,} 条记录")
                else:
                    print(f"    :warning:  清洗后无有效数据")

            except Exception as e:
                print(f"    :x: 加载失败: {e}")
                continue

        if all_data:
            # 合并所有数据
            combined_df = pd.concat(all_data, ignore_index=True)

            # 按时间排序
            combined_df = combined_df.sort_values('time').reset_index(drop=True)

            # 去重
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['latitude', 'longitude', 'time', 'magnitude'])
            after_dedup = len(combined_df)

            print(f"\n:bar_chart: 数据加载完成:")
            print(f"  总记录数: {after_dedup:,} 条")
            print(f"  去重移除: {before_dedup - after_dedup:,} 条")
            print(f"  时间范围: {combined_df['time'].min()} ~ {combined_df['time'].max()}")
            print(f"  震级范围: {combined_df['magnitude'].min():.1f} - {combined_df['magnitude'].max():.1f}")

            return combined_df
        else:
            print(":x: 未加载到任何有效数据")
            return pd.DataFrame()

    def map_to_grids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将地震数据映射到网格系统

        Args:
            df: 地震数据DataFrame

        Returns:
            添加网格信息的DataFrame
        """
        print("\n:japan: 将地震数据映射到网格...")

        # 计算网格索引
        grid_indices = []
        valid_count = 0

        for _, row in df.iterrows():
            lat_idx, lon_idx = self.grid_system.coords_to_grid(row['latitude'], row['longitude'])

            if lat_idx >= 0 and lon_idx >= 0:
                grid_indices.append((lat_idx, lon_idx))
                valid_count += 1
            else:
                grid_indices.append((-1, -1))

        # 添加网格信息
        df = df.copy()
        df['lat_idx'] = [idx[0] for idx in grid_indices]
        df['lon_idx'] = [idx[1] for idx in grid_indices]

        # 过滤有效网格
        df_valid = df[(df['lat_idx'] >= 0) & (df['lon_idx'] >= 0)].copy()

        # 统计网格分布
        grid_counts = df_valid.groupby(['lat_idx', 'lon_idx']).size()
        active_grids = len(grid_counts)

        print(f"  :white_check_mark: 映射完成:")
        print(f"    有效记录: {valid_count:,} / {len(df):,} ({valid_count/len(df)*100:.1f}%)")
        print(f"    活跃网格: {active_grids} / {self.grid_system.total_grids} ({active_grids/self.grid_system.total_grids*100:.1f}%)")
        print(f"    平均每网格: {valid_count/active_grids:.1f} 条记录")

        return df_valid