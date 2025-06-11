"""
独立的USGS数据抓取脚本
按年份保存原始数据，从最新数据开始倒序抓取
"""

import requests
import json
import csv
import io
from datetime import datetime, timedelta
import os
import time
import boto3
from botocore.exceptions import ClientError

class USGSDataFetcher:
    def __init__(self, s3_bucket=None, s3_prefix="earthquake-data/raw", local_mode=False):
        """
        初始化数据获取器
        
        Args:
            s3_bucket: S3存储桶名称，None表示本地模式
            s3_prefix: S3对象前缀
            local_mode: 是否使用本地模式
        """
        self.local_mode = local_mode or s3_bucket is None
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        
        if self.local_mode:
            self.data_dir = "../../../data"
            self.raw_dir = os.path.join(self.data_dir, "raw")
            os.makedirs(self.raw_dir, exist_ok=True)
        else:
            # 初始化S3客户端
            self.s3_client = boto3.client('s3')
            print(f"S3模式: 存储桶={s3_bucket}, 前缀={s3_prefix}")
        
        # 日本地理边界
        self.japan_bounds = {
            'min_lat': 24.0,   # 最南端（沖縄）
            'max_lat': 45.6,   # 最北端（北海道）
            'min_lon': 122.9,  # 最西端（沖縄）
            'max_lon': 146.0   # 最东端（北海道）
        }
    
    def fetch_earthquake_data(self, start_date, end_date, min_magnitude=3.0):
        """
        从USGS API获取地震数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            min_magnitude: 最小震级
            
        Returns:
            地震数据列表
        """
        print(f"正在获取 {start_date} 到 {end_date} 的地震数据...")
        
        params = {
            'format': 'geojson',
            'starttime': start_date,
            'endtime': end_date,
            'minmagnitude': min_magnitude,
            'minlatitude': self.japan_bounds['min_lat'],
            'maxlatitude': self.japan_bounds['max_lat'],
            'minlongitude': self.japan_bounds['min_lon'],
            'maxlongitude': self.japan_bounds['max_lon'],
            'limit': 20000  # USGS API限制
        }
        
        try:
            response = requests.get(
                'https://earthquake.usgs.gov/fdsnws/event/1/query',
                params=params,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                earthquakes = []
                
                for feature in data['features']:
                    props = feature['properties']
                    coords = feature['geometry']['coordinates']
                    
                    # 数据验证
                    if (props.get('mag') is not None and 
                        coords[0] is not None and 
                        coords[1] is not None):
                        
                        earthquake = {
                            'magnitude': props['mag'],
                            'longitude': coords[0],
                            'latitude': coords[1],
                            'depth': coords[2] if len(coords) > 2 else 0,
                            'time': datetime.fromtimestamp(props['time'] / 1000).isoformat(),
                            'place': props.get('place', ''),
                            'mag_type': props.get('magType', 'unknown'),
                            'id': props.get('ids', ''),
                            'url': props.get('url', '')
                        }
                        earthquakes.append(earthquake)
                
                print(f"获取到 {len(earthquakes)} 条地震记录")
                return earthquakes
                
            else:
                print(f"API请求失败: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"数据获取错误: {e}")
            return []
    
    def fetch_period_data(self, start_year, end_year, min_magnitude=3.0):
        """
        获取指定年份范围的所有地震数据（一次性获取5年）
        
        Args:
            start_year: 开始年份
            end_year: 结束年份
            min_magnitude: 最小震级
            
        Returns:
            该时间段所有地震数据列表
        """
        print(f"\n🗓️  开始获取 {start_year}-{end_year} 年数据...")
        
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        all_earthquakes = self.fetch_earthquake_data(start_date, end_date, min_magnitude)
        
        if all_earthquakes:
            # 保存数据
            filename = f"rawData_{start_year}-{end_year}.csv"
            
            if self.local_mode:
                # 本地保存
                filepath = os.path.join(self.raw_dir, filename)
                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['magnitude', 'longitude', 'latitude', 'depth', 'time', 
                                'place', 'mag_type', 'id', 'url']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_earthquakes)
                print(f"✅ {start_year}-{end_year}年数据已保存到本地: {filepath} ({len(all_earthquakes)} 条记录)")
            else:
                # S3保存
                if self.save_to_s3(all_earthquakes, filename):
                    print(f"✅ {start_year}-{end_year}年数据已保存到S3: {filename} ({len(all_earthquakes)} 条记录)")
                else:
                    print(f"❌ {start_year}-{end_year}年数据S3保存失败")
        else:
            print(f"⚠️  {start_year}-{end_year}年无数据")
        
        return all_earthquakes
    
    def fetch_all_historical_data(self, min_magnitude=3.0):
        """
        获取所有历史数据（按5年时间段从最新开始倒序）
        """
        print("🚀 开始获取USGS历史地震数据...")
        print(f"参数: 最小震级 {min_magnitude}")
        print(f"存储模式: {'本地' if self.local_mode else 'S3'}")
        print("数据范围: 日本地区 (纬度 {:.1f}-{:.1f}, 经度 {:.1f}-{:.1f})".format(
            self.japan_bounds['min_lat'], self.japan_bounds['max_lat'],
            self.japan_bounds['min_lon'], self.japan_bounds['max_lon']
        ))
        
        current_year = datetime.now().year
        start_year = 1970
        
        total_records = 0
        successful_periods = []
        
        # 按5年时间段从当前年份开始倒序获取
        for period_start in range(current_year, start_year - 1, -5):
            period_end = min(period_start + 4, current_year)
            
            try:
                # 检查文件是否已存在
                filename = f"rawData_{period_start}-{period_end}.csv"
                
                if self.local_mode:
                    # 本地文件检查
                    filepath = os.path.join(self.raw_dir, filename)
                    file_exists = os.path.exists(filepath)
                    if file_exists:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            existing_data = list(reader)
                        record_count = len(existing_data)
                else:
                    # S3文件检查
                    file_exists = self.check_s3_file_exists(filename)
                    if file_exists:
                        existing_data = self.load_from_s3(filename)
                        record_count = len(existing_data) if existing_data else 0

                if file_exists and record_count > 0:
                    print(f"📁 {period_start}-{period_end}年数据已存在: {record_count} 条记录")
                    total_records += record_count
                    successful_periods.append((period_start, period_end))
                    continue
                
                # 获取新数据
                period_data = self.fetch_period_data(period_start, period_end, min_magnitude)

                if period_data:  # 修改：检查列表是否为空，而不是 .empty
                    total_records += len(period_data)
                    successful_periods.append((period_start, period_end))

                # 时间段间隔（避免API限制）
                time.sleep(5)
                
            except Exception as e:
                print(f"❌ {period_start}-{period_end}年数据获取失败: {e}")
                continue
        
        print("\n" + "="*50)
        print("📊 数据获取完成!")
        print("="*50)
        print(f"成功获取时间段: {len(successful_periods)} 个")
        if successful_periods:
            all_years = []
            for start, end in successful_periods:
                all_years.extend(range(start, end + 1))
            print(f"年份范围: {min(all_years)} - {max(all_years)}")
        print(f"总记录数: {total_records:,} 条")
        storage_location = self.raw_dir if self.local_mode else f"s3://{self.s3_bucket}/{self.s3_prefix}"
        print(f"数据保存位置: {storage_location}")
    
        return successful_periods, total_records  
    
    def save_to_s3(self, data_list, filename):
        """
        保存数据列表到S3为CSV格式
        
        Args:
            data_list: 地震数据列表
            filename: 文件名
            
        Returns:
            bool: 保存是否成功
        """
        try:
            if not data_list:
                print(f"⚠️ 数据为空，跳过保存: {filename}")
                return False
            
            # 创建CSV字符串
            csv_buffer = io.StringIO()
            
            # 写入CSV头部
            fieldnames = ['magnitude', 'longitude', 'latitude', 'depth', 'time', 
                        'place', 'mag_type', 'id', 'url']
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()
            
            # 写入数据
            for earthquake in data_list:
                writer.writerow(earthquake)
            
            csv_string = csv_buffer.getvalue()
            
            # 构建S3对象键
            s3_key = f"{self.s3_prefix}/{filename}"
            
            # 上传到S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=csv_string.encode('utf-8'),
                ContentType='text/csv'
            )
            
            print(f"✅ 文件已保存到S3: s3://{self.s3_bucket}/{s3_key}")
            return True
            
        except ClientError as e:
            print(f"❌ S3保存失败: {e}")
            return False

    def load_from_s3(self, filename):
        """
        从S3加载CSV文件
        
        Args:
            filename: 文件名
            
        Returns:
            数据列表或None
        """
        try:
            s3_key = f"{self.s3_prefix}/{filename}"
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            csv_string = response['Body'].read().decode('utf-8')
            
            # 解析CSV
            csv_reader = csv.DictReader(io.StringIO(csv_string))
            data_list = []
            for row in csv_reader:
                # 转换数值类型
                try:
                    row['magnitude'] = float(row['magnitude'])
                    row['longitude'] = float(row['longitude'])
                    row['latitude'] = float(row['latitude'])
                    row['depth'] = float(row['depth'])
                except (ValueError, TypeError):
                    continue
                data_list.append(row)
            
            return data_list
            
        except ClientError as e:
            print(f"❌ 从S3读取失败: {e}")
            return None

    def check_s3_file_exists(self, filename):
        """
        检查S3中文件是否存在
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 文件是否存在
        """
        try:
            s3_key = f"{self.s3_prefix}/{filename}"
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
            return True
        except ClientError:
            return False

def main(mode):
    """主函数"""
    print("USGS日本地震数据抓取系统")
    print("支持本地模式和S3模式")
    
    # 选择运行模式 1: 本地模式, 2: S3模式
    if mode == 1:
        # 本地模式
        print("使用本地模式...")
        fetcher = USGSDataFetcher(local_mode=True)
    elif mode == 2:
        # S3模式
        s3_bucket = "earthquake-prediction-shuhao"
        if not s3_bucket:
            print("❌ 必须提供S3存储桶名称")
            return
        
        s3_prefix = input("输入S3前缀 (默认: earthquake-data/raw): ").strip()
        if not s3_prefix:
            s3_prefix = "earthquake-data/raw"
        
        print(f"使用S3模式: {s3_bucket}/{s3_prefix}")
        fetcher = USGSDataFetcher(s3_bucket=s3_bucket, s3_prefix=s3_prefix)
    else:
        print("❌ 无效选择")
        return
    
    min_magnitude = 3.0  # 最小震级
    
    # 开始抓取
    successful_periods, total_records = fetcher.fetch_all_historical_data(min_magnitude)
    
    print(f"\n🎉 数据抓取完成! 共获取 {total_records:,} 条记录")

if __name__ == "__main__":
    main(1)