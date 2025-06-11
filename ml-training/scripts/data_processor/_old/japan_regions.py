"""
日本都道府县地理边界定义
基于经纬度范围进行简化划分
"""

# 日本47都道府县的地理边界 (简化版本)
JAPAN_PREFECTURES = {
    "北海道": {
        "lat_range": (41.0, 45.6),
        "lon_range": (139.0, 146.0),
        "center": (43.3, 142.5)
    },
    "青森県": {
        "lat_range": (40.2, 41.6),
        "lon_range": (139.5, 141.7),
        "center": (40.8, 140.6)
    },
    "岩手県": {
        "lat_range": (38.7, 40.4),
        "lon_range": (140.3, 142.1),
        "center": (39.5, 141.2)
    },
    "宮城県": {
        "lat_range": (37.7, 39.0),
        "lon_range": (140.3, 141.7),
        "center": (38.4, 141.0)
    },
    "秋田県": {
        "lat_range": (38.8, 40.7),
        "lon_range": (139.4, 141.2),
        "center": (39.7, 140.3)
    },
    "山形県": {
        "lat_range": (37.7, 39.0),
        "lon_range": (139.5, 140.6),
        "center": (38.3, 140.0)
    },
    "福島県": {
        "lat_range": (36.8, 38.0),
        "lon_range": (139.0, 141.1),
        "center": (37.4, 140.0)
    },
    "茨城県": {
        "lat_range": (35.7, 36.9),
        "lon_range": (139.6, 140.9),
        "center": (36.3, 140.3)
    },
    "栃木県": {
        "lat_range": (36.2, 37.0),
        "lon_range": (139.0, 140.3),
        "center": (36.6, 139.7)
    },
    "群馬県": {
        "lat_range": (36.0, 36.9),
        "lon_range": (138.4, 139.9),
        "center": (36.4, 139.1)
    },
    "埼玉県": {
        "lat_range": (35.7, 36.3),
        "lon_range": (138.7, 140.0),
        "center": (36.0, 139.3)
    },
    "千葉県": {
        "lat_range": (34.9, 36.1),
        "lon_range": (139.7, 140.9),
        "center": (35.5, 140.3)
    },
    "東京都": {
        "lat_range": (35.5, 35.9),
        "lon_range": (139.0, 140.0),
        "center": (35.7, 139.5)
    },
    "神奈川県": {
        "lat_range": (35.1, 35.7),
        "lon_range": (139.0, 139.8),
        "center": (35.4, 139.4)
    },
    "新潟県": {
        "lat_range": (36.7, 38.6),
        "lon_range": (137.6, 139.9),
        "center": (37.6, 138.8)
    },
    "富山県": {
        "lat_range": (36.3, 36.9),
        "lon_range": (136.8, 137.9),
        "center": (36.6, 137.4)
    },
    "石川県": {
        "lat_range": (36.0, 37.9),
        "lon_range": (136.2, 137.4),
        "center": (36.9, 136.8)
    },
    "福井県": {
        "lat_range": (35.3, 36.4),
        "lon_range": (135.4, 136.9),
        "center": (35.8, 136.2)
    },
    "山梨県": {
        "lat_range": (35.1, 35.9),
        "lon_range": (138.2, 139.2),
        "center": (35.5, 138.7)
    },
    "長野県": {
        "lat_range": (35.0, 37.0),
        "lon_range": (137.3, 138.9),
        "center": (36.0, 138.1)
    },
    "岐阜県": {
        "lat_range": (35.2, 36.4),
        "lon_range": (136.2, 137.8),
        "center": (35.8, 137.0)
    },
    "静岡県": {
        "lat_range": (34.6, 35.7),
        "lon_range": (137.4, 139.2),
        "center": (35.1, 138.3)
    },
    "愛知県": {
        "lat_range": (34.6, 35.4),
        "lon_range": (136.7, 137.8),
        "center": (35.0, 137.3)
    },
    "三重県": {
        "lat_range": (33.7, 35.2),
        "lon_range": (135.8, 136.9),
        "center": (34.5, 136.4)
    },
    "滋賀県": {
        "lat_range": (34.7, 35.7),
        "lon_range": (135.7, 136.4),
        "center": (35.2, 136.1)
    },
    "京都府": {
        "lat_range": (34.7, 35.8),
        "lon_range": (135.0, 136.0),
        "center": (35.3, 135.5)
    },
    "大阪府": {
        "lat_range": (34.3, 34.8),
        "lon_range": (135.2, 135.8),
        "center": (34.6, 135.5)
    },
    "兵庫県": {
        "lat_range": (34.6, 35.7),
        "lon_range": (134.2, 135.5),
        "center": (35.1, 134.8)
    },
    "奈良県": {
        "lat_range": (33.9, 34.8),
        "lon_range": (135.6, 136.1),
        "center": (34.4, 135.9)
    },
    "和歌山県": {
        "lat_range": (33.4, 34.4),
        "lon_range": (135.0, 136.0),
        "center": (33.9, 135.5)
    },
    "鳥取県": {
        "lat_range": (35.0, 35.7),
        "lon_range": (133.2, 134.3),
        "center": (35.4, 133.8)
    },
    "島根県": {
        "lat_range": (34.0, 35.8),
        "lon_range": (131.7, 133.4),
        "center": (34.9, 132.6)
    },
    "岡山県": {
        "lat_range": (34.2, 35.3),
        "lon_range": (133.2, 134.4),
        "center": (34.8, 133.8)
    },
    "広島県": {
        "lat_range": (34.0, 34.9),
        "lon_range": (132.2, 133.3),
        "center": (34.4, 132.8)
    },
    "山口県": {
        "lat_range": (33.7, 34.6),
        "lon_range": (130.8, 132.2),
        "center": (34.2, 131.5)
    },
    "徳島県": {
        "lat_range": (33.4, 34.4),
        "lon_range": (133.5, 134.8),
        "center": (33.9, 134.2)
    },
    "香川県": {
        "lat_range": (34.1, 34.5),
        "lon_range": (133.4, 134.5),
        "center": (34.3, 134.0)
    },
    "愛媛県": {
        "lat_range": (32.8, 34.4),
        "lon_range": (132.3, 133.3),
        "center": (33.6, 132.8)
    },
    "高知県": {
        "lat_range": (32.7, 33.9),
        "lon_range": (132.4, 134.3),
        "center": (33.3, 133.4)
    },
    "福岡県": {
        "lat_range": (33.0, 33.9),
        "lon_range": (129.7, 131.1),
        "center": (33.5, 130.4)
    },
    "佐賀県": {
        "lat_range": (33.0, 33.5),
        "lon_range": (129.7, 130.4),
        "center": (33.3, 130.1)
    },
    "長崎県": {
        "lat_range": (32.6, 34.7),
        "lon_range": (128.8, 130.4),
        "center": (33.6, 129.6)
    },
    "熊本県": {
        "lat_range": (32.0, 33.3),
        "lon_range": (130.2, 131.3),
        "center": (32.7, 130.8)
    },
    "大分県": {
        "lat_range": (32.8, 33.8),
        "lon_range": (130.8, 132.0),
        "center": (33.3, 131.4)
    },
    "宮崎県": {
        "lat_range": (31.4, 32.9),
        "lon_range": (130.7, 132.0),
        "center": (32.2, 131.4)
    },
    "鹿児島県": {
        "lat_range": (24.0, 32.1),
        "lon_range": (128.4, 131.5),
        "center": (28.0, 130.0)
    },
    "沖縄県": {
        "lat_range": (24.0, 26.9),
        "lon_range": (122.9, 131.3),
        "center": (25.5, 127.1)
    }
}

def get_prefecture_by_coordinates(latitude, longitude):
    """
    根据经纬度判断属于哪个都道府县
    """
    for prefecture, bounds in JAPAN_PREFECTURES.items():
        lat_min, lat_max = bounds["lat_range"]
        lon_min, lon_max = bounds["lon_range"]
        
        if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
            return prefecture
    
    # 如果不在任何定义范围内，返回最近的
    min_distance = float('inf')
    closest_prefecture = None
    
    for prefecture, bounds in JAPAN_PREFECTURES.items():
        center_lat, center_lon = bounds["center"]
        distance = ((latitude - center_lat) ** 2 + (longitude - center_lon) ** 2) ** 0.5
        
        if distance < min_distance:
            min_distance = distance
            closest_prefecture = prefecture
    
    return closest_prefecture

def get_all_prefecture_names():
    """获取所有都道府县名称"""
    return list(JAPAN_PREFECTURES.keys())