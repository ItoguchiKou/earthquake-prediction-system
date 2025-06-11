"""
Lambda函数入口点
"""

from fetch_raw_data import USGSDataFetcher
import os

def lambda_handler(event, context):
    """
    Lambda函数入口点
    
    Args:
        event: Lambda事件数据
        context: Lambda上下文
        
    Returns:
        响应数据
    """
    try:
        # 从事件中获取参数
        s3_bucket = event.get('s3_bucket', os.environ.get('S3_BUCKET'))
        s3_prefix = event.get('s3_prefix', 'earthquake-data/raw')
        min_magnitude = event.get('min_magnitude', 3.0)
        
        if not s3_bucket:
            return {
                'statusCode': 400,
                'body': 'S3_BUCKET environment variable or s3_bucket parameter is required'
            }
        
        print(f"Lambda执行开始: bucket={s3_bucket}, prefix={s3_prefix}, min_mag={min_magnitude}")
        
        # 初始化数据获取器
        fetcher = USGSDataFetcher(s3_bucket=s3_bucket, s3_prefix=s3_prefix)
        
        # 获取数据
        successful_periods, total_records = fetcher.fetch_all_historical_data(min_magnitude)
        
        return {
            'statusCode': 200,
            'body': {
                'message': '数据获取完成',
                'successful_periods': len(successful_periods),
                'total_records': total_records,
                's3_bucket': s3_bucket,
                's3_prefix': s3_prefix
            }
        }
        
    except Exception as e:
        print(f"❌ Lambda执行失败: {str(e)}")
        return {
            'statusCode': 500,
            'body': f'Error: {str(e)}'
        }