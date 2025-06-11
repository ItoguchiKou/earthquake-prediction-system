// 日本地震预测系统固定网格配置
const JAPAN_GRID_CONFIG = {

    
    grids: {
        '1': {
            id: '1',
            lat: [24, 27],
            lon: [126, 129],
            prefectures: ['沖縄県'],
            region: '沖縄'
        },
        '2': {
            id: '2',
            lat: [27, 30],
            lon: [129, 132],
            prefectures: ['鹿児島県（南部離島）'],
            region: '南西諸島'
        },
        '3': {
            id: '3',
            lat: [30, 33],
            lon: [129, 132],
            prefectures: ['長崎県', '佐賀県', '熊本県'],
            region: '九州西部'
        },
        '4': {
            id: '4',
            lat: [30, 33],
            lon: [132, 135],
            prefectures: ['大分県', '宮崎県', '鹿児島県（本土）'],
            region: '九州東部'
        },
        '5': {
            id: '5',
            lat: [33, 36],
            lon: [129, 132],
            prefectures: ['福岡県', '山口県'],
            region: '九州北部・中国西端'
        },
        '6': {
            id: '6',
            lat: [33, 36],
            lon: [132, 135],
            prefectures: ['広島県', '愛媛県', '高知県（西部）'],
            region: '中国・四国西部'
        },
        '7': {
            id: '7',
            lat: [33, 36],
            lon: [135, 138],
            prefectures: ['大阪府', '兵庫県', '京都府（南部）', '奈良県', '和歌山県', '香川県', '徳島県'],
            region: '近畿・四国東部'
        },
        '8': {
            id: '8',
            lat: [33, 36],
            lon: [138, 141],
            prefectures: ['東京都', '神奈川県', '千葉県', '埼玉県', '山梨県', '静岡県（東部）'],
            region: '関東南部'
        },
        '9': {
            id: '9',
            lat: [36, 39],
            lon: [135, 138],
            prefectures: ['石川県', '福井県', '滋賀県', '京都府（北部）', '岐阜県（西部）'],
            region: '北陸'
        },
        '10': {
            id: '10',
            lat: [36, 39],
            lon: [138, 141],
            prefectures: ['長野県', '群馬県', '栃木県', '茨城県（西部）', '新潟県（南部）', '岐阜県（東部）', '愛知県', '静岡県（西部）'],
            region: '中部・関東北部'
        },
        '11': {
            id: '11',
            lat: [36, 39],
            lon: [141, 144],
            prefectures: ['茨城県（東部）', '福島県（南部）'],
            region: '関東北東部'
        },
        '12': {
            id: '12',
            lat: [39, 42],
            lon: [138, 141],
            prefectures: ['新潟県（北部）', '山形県', '秋田県（南部）', '富山県'],
            region: '東北日本海側'
        },
        '13': {
            id: '13',
            lat: [39, 42],
            lon: [141, 144],
            prefectures: ['宮城県', '福島県（北部）', '岩手県（南部）'],
            region: '東北太平洋側'
        },
        '14': {
            id: '14',
            lat: [42, 45],
            lon: [138, 141],
            prefectures: ['北海道（道西・道央西部）', '青森県（西部）', '秋田県（北部）'],
            region: '北海道西部・東北北西部'
        },
        '15': {
            id: '15',
            lat: [42, 45],
            lon: [141, 144],
            prefectures: ['北海道（道央・道南）', '青森県（東部）', '岩手県（北部）'],
            region: '北海道中部・東北北東部'
        },
        '16': {
            id: '16',
            lat: [42, 45],
            lon: [144, 147],
            prefectures: ['北海道（道東）'],
            region: '北海道東部'
        }
    },
    
    // 获取所有网格ID列表
    getGridIds() {
        return Object.keys(this.grids);
    },
    
    // 获取网格信息
    getGridInfo(gridId) {
        return this.grids[gridId];
    },
    
    // 获取所有网格信息
    getAllGrids() {
        return Object.values(this.grids);
    }
};

// 冻结配置对象，防止意外修改
Object.freeze(JAPAN_GRID_CONFIG);
Object.freeze(JAPAN_GRID_CONFIG.grids);