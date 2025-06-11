// 数据管理模块
class DataManager {
    constructor() {
        this.data = null;
        this.currentTimeWindow = 7;
        this.apiUrl = 'http://18.179.22.39:8080/api'; // TODO: 替换为实际的EC2地址
    }

    // 获取预测数据
    async fetchPredictions() {
        try {
            // TODO: 正式环境下使用真实API
            const response = await fetch(`${this.apiUrl}/predictions`);
            const data = await response.json();
            this.data = data;
            
            // 使用模拟数据
            // this.data = MOCK_DATA;
            // this.saveToLocalStorage(this.data);
            
            return this.data;
        } catch (error) {
            console.error('Failed to fetch predictions:', error);
            // 尝试从本地缓存加载
            return this.loadFromLocalStorage();
        }
    }

    // 保存到本地存储
    saveToLocalStorage(data) {
        try {
            localStorage.setItem('earthquakePredictions', JSON.stringify(data));
            localStorage.setItem('lastUpdate', new Date().toISOString());
        } catch (e) {
            console.warn('Failed to save to localStorage:', e);
        }
    }

    // 从本地存储加载
    loadFromLocalStorage() {
        try {
            const data = localStorage.getItem('earthquakePredictions');
            return data ? JSON.parse(data) : null;
        } catch (e) {
            console.warn('Failed to load from localStorage:', e);
            return null;
        }
    }

    // 获取指定时间窗口的数据（结合固定网格配置）
    getDataForTimeWindow(timeWindow) {
        if (!this.data) return null;
        
        const timeKey = `${timeWindow}days`;
        const gridData = [];
        
        // 遍历所有固定网格
        JAPAN_GRID_CONFIG.getAllGrids().forEach(gridConfig => {
            const prediction = this.data.predictions[gridConfig.id];
            if (prediction) {
                // 合并网格配置和预测数据
                gridData.push({
                    ...gridConfig,
                    currentProbability: prediction.probabilities[timeKey],
                    riskLevel: prediction.probabilities[timeKey].riskLevel
                });
            }
        });
        
        return gridData;
    }

    // 获取最高风险值
    getMaxRisk() {
        if (!this.data) return 0;
        
        const timeKey = `${this.currentTimeWindow}days`;
        let maxRisk = 0;
        
        Object.values(this.data.predictions).forEach(prediction => {
            const risk = prediction.probabilities[timeKey].combined;
            if (risk > maxRisk) maxRisk = risk;
        });
        
        return maxRisk;
    }

    // 格式化更新时间
    formatUpdateTime(isoString) {
        const date = new Date(isoString);
        const month = (date.getMonth() + 1).toString().padStart(2, '0');
        const day = date.getDate().toString().padStart(2, '0');
        const hours = date.getHours().toString().padStart(2, '0');
        const minutes = date.getMinutes().toString().padStart(2, '0');
        
        return `${month}/${day} ${hours}:${minutes}`;
    }
}

// 创建全局数据管理器实例
const dataManager = new DataManager();