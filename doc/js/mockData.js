// 模拟数据生成器 - 只生成概率数据
const MockDataGenerator = {
    // 生成随机概率值
    generateProbability(baseProb, variance = 0.1) {
        const prob = baseProb + (Math.random() - 0.5) * variance;
        return Math.max(0, Math.min(1, prob));
    },

    // 根据当前时间窗口的概率计算风险等级
    getRiskLevel(probability) {
        if (probability < 0.1) return 'low';      // 10%以下 - 低
        if (probability < 0.3) return 'medium';   // 10-30% - 中
        if (probability < 0.5) return 'high';     // 30-50% - 高
        return 'extreme';                         // 50%以上 - 极高
    },

    // 生成单个网格的预测数据（只包含概率）
    generateGridPrediction(gridId) {
        const gridInfo = JAPAN_GRID_CONFIG.getGridInfo(gridId);
        if (!gridInfo) return null;
        
        // 基础概率（东部沿海地区概率更高）
        const lonCenter = (gridInfo.lon[0] + gridInfo.lon[1]) / 2;
        const latCenter = (gridInfo.lat[0] + gridInfo.lat[1]) / 2;
        
        // 根据地理位置设置基础风险
        let baseRisk = 0.05;  // 降低基础风险
        
        // 关东地区（网格11）风险最高
        if (gridId === '11') {
            baseRisk = 0.15;
        }
        // 东北太平洋侧（网格13）
        else if (gridId === '13') {
            baseRisk = 0.12;
        }
        // 近畿地区（网格8）
        else if (gridId === '8') {
            baseRisk = 0.10;
        }
        // 其他太平洋沿岸
        else if (lonCenter >= 140) {
            baseRisk = 0.08;
        }
        // 内陆地区
        else {
            baseRisk = 0.03;
        }
        
        // 为不同时间窗口生成概率
        const predictions = {};
        const timeWindows = [7, 14, 30];
        
        const riskLevels = {};
        
        timeWindows.forEach(days => {
            const timeMultiplier = Math.sqrt(days / 7) * 0.5; // 时间越长，概率适度增加
            
            const m3_45 = this.generateProbability(baseRisk * 0.8 + timeMultiplier * 0.05);
            const m45_55 = this.generateProbability(baseRisk * 0.3 + timeMultiplier * 0.02);
            const m55_65 = this.generateProbability(baseRisk * 0.1 + timeMultiplier * 0.01);
            const m65_plus = this.generateProbability(baseRisk * 0.02 + timeMultiplier * 0.002);
            
            const combined = Math.max(m3_45, m45_55, m55_65, m65_plus);
            
            predictions[`${days}days`] = {
                'M3_4.5': m3_45,
                'M4.5_5.5': m45_55,
                'M5.5_6.5': m55_65,
                'M6.5+': m65_plus,
                'combined': combined
            };
            
            // 为每个时间窗口计算风险等级
            riskLevels[`${days}days`] = this.getRiskLevel(combined);
        });
        
        return {
            gridId: gridId,
            probabilities: predictions,
            riskLevels: riskLevels  // 为每个时间窗口保存风险等级
        };
    },

    // 生成模拟数据（符合后端API格式）
    generateMockData() {
        const predictions = {};
        
        // 为每个固定网格生成预测数据
        JAPAN_GRID_CONFIG.getGridIds().forEach(gridId => {
            const prediction = this.generateGridPrediction(gridId);
            if (prediction) {
                predictions[gridId] = prediction;
            }
        });
        
        return {
            updateTime: new Date().toISOString(),
            predictions: predictions
        };
    }
};

// 生成模拟数据
const MOCK_DATA = MockDataGenerator.generateMockData();