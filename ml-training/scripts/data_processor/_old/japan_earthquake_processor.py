"""
日本地震数据处理主脚本
从USGS API获取数据并按都道府县分类保存
"""

import pandas as pd
import os
import glob
from japan_regions import get_prefecture_by_coordinates, get_all_prefecture_names
from data_features import create_earthquake_features, prepare_training_data

class JapanEarthquakeProcessor:
    def __init__(self, data_dir="../../../data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # 创建目录
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_raw_data(self, years=None, min_magnitude=3.0):
        """
        从本地文件加载原始地震数据
        
        Args:
            years: 指定年份列表，None表示加载所有年份
            min_magnitude: 最小震级过滤
            
        Returns:
            合并后的地震数据DataFrame
        """
        print("📂 正在加载本地原始数据...")
        
        # 查找所有原始数据文件（支持两种命名格式）
        pattern1 = os.path.join(self.raw_dir, "rawData_*.csv")
        raw_files = glob.glob(pattern1)
        
        if not raw_files:
            print(f"❌ 在 {self.raw_dir} 目录下未找到原始数据文件")
            print("请先运行 fetch_raw_data.py 获取原始数据")
            return pd.DataFrame()
        
        all_data = []
        loaded_years = []
        
        for file_path in sorted(raw_files):
            filename = os.path.basename(file_path)
            year_part = filename.replace('rawData_', '').replace('.csv', '')
            
            try:
                # 处理年份范围格式 (如 2020-2024) 和单年份格式 (如 2024)
                if '-' in year_part:
                    start_year, end_year = map(int, year_part.split('-'))
                    file_years = list(range(start_year, end_year + 1))
                else:
                    file_years = [int(year_part)]
                
                # 如果指定了年份，检查是否有交集
                if years is not None:
                    file_years = [y for y in file_years if y in years]
                    if not file_years:
                        continue
                
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                
                if not df.empty:
                    # 如果指定了年份且文件包含多年数据，按年份过滤
                    if years is not None and '-' in year_part:
                        df['time'] = pd.to_datetime(df['time'])
                        df = df[df['time'].dt.year.isin(years)]
                    
                    # 数据清洗
                    df = df.dropna(subset=['latitude', 'longitude', 'magnitude'])
                    df = df[df['magnitude'] >= min_magnitude]
                    
                    if not df.empty:
                        all_data.append(df)
                        loaded_years.extend(file_years)
                        print(f"  ✅ {year_part}: {len(df)} 条记录")
                    else:
                        print(f"  ⚠️  {year_part}: 清洗后无有效数据")
                else:
                    print(f"  ⚠️  {year_part}: 文件为空")
                    
            except ValueError:
                print(f"  ❌ 无法解析年份: {filename}")
                continue
            except Exception as e:
                print(f"  ❌ 加载 {filename} 失败: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # 确保时间列是datetime类型
            combined_df['time'] = pd.to_datetime(combined_df['time'])
            
            # 按时间排序
            combined_df = combined_df.sort_values('time').reset_index(drop=True)
            
            # 去重（去除年份范围重叠可能导致的重复数据）
            loaded_years = sorted(set(loaded_years))
            
            print(f"\n📊 数据加载完成:")
            print(f"  年份范围: {min(loaded_years)} - {max(loaded_years)}")
            print(f"  总记录数: {len(combined_df):,} 条")
            print(f"  时间范围: {combined_df['time'].min()} ~ {combined_df['time'].max()}")
            
            return combined_df
        else:
            print("❌ 未加载到任何有效数据")
            return pd.DataFrame()
    
    def assign_prefectures(self, df):
        """
        为地震数据分配都道府县
        """
        print("🗾 正在分配都道府県...")
        
        prefectures = []
        for _, row in df.iterrows():
            prefecture = get_prefecture_by_coordinates(row['latitude'], row['longitude'])
            prefectures.append(prefecture)
        
        df['prefecture'] = prefectures
        return df
    
    def save_by_prefecture(self, df):
        """
        按都道府县保存数据
        """
        print("💾 正在按都道府県保存数据...")
        
        prefecture_stats = {}
        
        for prefecture in get_all_prefecture_names():
            prefecture_data = df[df['prefecture'] == prefecture].copy()
            
            if len(prefecture_data) > 0:
                # 创建特征
                features_df = create_earthquake_features(prefecture_data)
                
                if len(features_df) > 0:
                    # 保存原始数据
                    raw_file = os.path.join(self.processed_dir, f"{prefecture}_地震データ.csv")
                    prefecture_data.to_csv(raw_file, index=False, encoding='utf-8-sig')
                    
                    # 保存特征数据
                    features_file = os.path.join(self.processed_dir, f"{prefecture}_特征データ.csv")
                    features_df.to_csv(features_file, index=False, encoding='utf-8-sig')
                    
                    # 准备训练数据
                    X, y = prepare_training_data(features_df)
                    
                    prefecture_stats[prefecture] = {
                        'earthquake_count': len(prefecture_data),
                        'feature_count': len(features_df),
                        'trainable': X is not None,
                        'avg_magnitude': prefecture_data['magnitude'].mean(),
                        'max_magnitude': prefecture_data['magnitude'].max(),
                        'date_range': f"{prefecture_data['time'].min()} ~ {prefecture_data['time'].max()}"
                    }
                    
                    print(f"  ✅ {prefecture}: {len(prefecture_data)} 条地震记录, {len(features_df)} 条特征记录")
                else:
                    print(f"  ⚠️  {prefecture}: {len(prefecture_data)} 条地震记录, 但无法生成特征")
                    prefecture_stats[prefecture] = {
                        'earthquake_count': len(prefecture_data),
                        'feature_count': 0,
                        'trainable': False,
                        'avg_magnitude': prefecture_data['magnitude'].mean(),
                        'max_magnitude': prefecture_data['magnitude'].max(),
                        'date_range': f"{prefecture_data['time'].min()} ~ {prefecture_data['time'].max()}"
                    }
        
        # 保存统计信息
        stats_df = pd.DataFrame(prefecture_stats).T
        stats_file = os.path.join(self.processed_dir, "prefecture_statistics.csv")
        stats_df.to_csv(stats_file, encoding='utf-8-sig')
        
        print(f"\n📊 数据处理完成! 统计信息已保存到: {stats_file}")
        return prefecture_stats
    
    def process_all_data(self, years=None, min_magnitude=3.0):
        """
        完整的数据处理流程
        """
        print("🚀 开始日本地震数据处理流程...")
        print(f"参数: 最小震级{min_magnitude}")
        if years:
            print(f"指定年份: {years}")
        
        # 1. 加载原始数据
        df = self.load_raw_data(years, min_magnitude)
        
        if df.empty:
            print("❌ 未加载到数据，请先运行 fetch_raw_data.py 获取原始数据")
            return None, None
        
        # 2. 数据清洗
        print(f"加载数据: {len(df)} 条记录")
        df = df.dropna(subset=['latitude', 'longitude', 'magnitude'])
        df = df[df['magnitude'] >= min_magnitude]
        print(f"清洗后数据: {len(df)} 条记录")
        
        # 3. 分配都道府县
        df = self.assign_prefectures(df)
        
        # 4. 按都道府县保存
        stats = self.save_by_prefecture(df)
        
        # 5. 打印统计摘要
        self.print_summary(stats)
        
        print("🎉 数据处理完成!")
        
        return df, stats
    
    def print_summary(self, stats):
        """打印处理结果摘要"""
        print("\n" + "="*50)
        print("📊 处理结果摘要")
        print("="*50)
        
        total_earthquakes = sum(s['earthquake_count'] for s in stats.values())
        trainable_prefectures = sum(1 for s in stats.values() if s['trainable'])
        
        print(f"总地震记录数: {total_earthquakes:,}")
        print(f"涉及都道府県: {len(stats)}")
        print(f"可训练都道府県: {trainable_prefectures}")
        
        print("\n🏆 地震记录最多的都道府県:")
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['earthquake_count'], reverse=True)
        for i, (prefecture, stat) in enumerate(sorted_stats[:10]):
            print(f"{i+1:2d}. {prefecture}: {stat['earthquake_count']:4d} 条记录")
        
        print(f"\n📁 数据文件保存在: {self.processed_dir}")

def main():
    """主函数"""
    processor = JapanEarthquakeProcessor()
    
    # 处理参数
    min_magnitude = 3.0  # 最小震级3.0
    
    print("日本地震数据处理系统")
    print(f"将处理本地原始数据，震级≥{min_magnitude}")
    print("默认处理所有可用年份数据（通常为最近5年）")
    
    # 可选：指定处理特定年份
    # years = [2022, 2023, 2024]  # 取消注释可指定年份
    years = None  # 处理所有可用年份
    
    # 开始处理
    df, stats = processor.process_all_data(years, min_magnitude)

if __name__ == "__main__":
    main()