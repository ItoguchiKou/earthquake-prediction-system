// 地图管理模块
class MapManager {
    constructor() {
        this.map = null;
        this.gridLayers = new Map();
        this.currentTimeWindow = 7;
    }

    // 初始化地图
    initMap() {
        // 创建地图实例，中心点设在日本中部，调整缩放级别
        this.map = L.map('map', {
            center: [37.5, 137.5],  // 调整中心点
            zoom: 5.5,              // 调整初始缩放
            minZoom: 5,
            maxZoom: 8,
            zoomControl: false,
            maxBounds: [            // 限制地图边界
                [20, 120],          // 西南角
                [50, 150]           // 东北角
            ]
        });

        // 添加地图瓦片层
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);

        // 添加缩放控件到右下角
        L.control.zoom({
            position: 'bottomright'
        }).addTo(this.map);

        // 禁用双击缩放（移动端体验）
        this.map.doubleClickZoom.disable();
    }

    // 根据概率值获取颜色
    getProbabilityColor(probability) {
        // 概率值从0到1，映射到颜色
        if (probability < 0.1) return '#4CAF50';      // 绿色 - 低风险
        if (probability < 0.3) return '#FFC107';      // 黄色 - 中风险
        if (probability < 0.5) return '#FF9800';      // 橙色 - 高风险
        return '#F44336';                             // 红色 - 极高风险
    }

    // 创建网格层
    createGridLayer(gridData) {
        const bounds = [
            [gridData.lat[0], gridData.lon[0]],
            [gridData.lat[1], gridData.lon[1]]
        ];

        const probability = gridData.currentProbability.combined;
        const color = this.getProbabilityColor(probability);

        // 创建矩形网格
        const rectangle = L.rectangle(bounds, {
            color: '#666',          // 边框颜色调暗
            weight: 1,
            fillColor: color,
            fillOpacity: 0.6,
            className: 'grid-cell'
        });

        // 添加概率标签
        const centerLat = (gridData.lat[0] + gridData.lat[1]) / 2;
        const centerLon = (gridData.lon[0] + gridData.lon[1]) / 2;
        
        const label = L.divIcon({
            html: `<div class="grid-label">${(probability * 100).toFixed(0)}%</div>`,
            className: 'grid-label-icon',
            iconSize: [40, 20]
        });

        const marker = L.marker([centerLat, centerLon], {
            icon: label,
            interactive: false
        });

        // 绑定点击事件 - 传递完整的网格数据
        rectangle.on('click', () => {
            this.showGridDetails(gridData);
        });

        // 添加悬停效果
        rectangle.on('mouseover', function() {
            this.setStyle({ weight: 2, color: '#333' });
        });
        
        rectangle.on('mouseout', function() {
            this.setStyle({ weight: 1, color: '#666' });
        });

        // 保存到图层组
        const layerGroup = L.layerGroup([rectangle, marker]);
        this.gridLayers.set(gridData.id, {
            group: layerGroup,
            data: gridData
        });

        return layerGroup;
    }

    // 更新所有网格
    updateGrids(predictions) {
        // 清除现有网格
        this.gridLayers.forEach(layer => {
            layer.group.remove();
        });
        this.gridLayers.clear();

        // 添加新网格
        predictions.forEach(gridData => {
            const layer = this.createGridLayer(gridData);
            layer.addTo(this.map);
        });
    }

    // 更新网格颜色（切换时间窗口时）
    updateGridColors(timeWindow) {
        this.currentTimeWindow = timeWindow;

        this.gridLayers.forEach((layer, gridId) => {
            const gridData = layer.data;
            
            // 从全局数据中获取新的概率数据
            const newData = dataManager.data?.predictions[gridId];
            if (!newData || !newData.probabilities) {
                console.warn(`No probability data for grid ${gridId}`);
                return;
            }
            
            const probability = newData.probabilities[`${timeWindow}days`].combined;
            const color = this.getProbabilityColor(probability);

            // 更新矩形颜色
            const rectangle = layer.group.getLayers()[0];
            rectangle.setStyle({
                fillColor: color
            });

            // 更新标签
            const marker = layer.group.getLayers()[1];
            const newLabel = L.divIcon({
                html: `<div class="grid-label">${(probability * 100).toFixed(0)}%</div>`,
                className: 'grid-label-icon',
                iconSize: [40, 20]
            });
            marker.setIcon(newLabel);
            
            // 更新layer中保存的数据
            layer.data.currentProbability = newData.probabilities[`${timeWindow}days`];
        });
    }

    showGridDetails(gridData) {
        const modal = document.getElementById('gridModal');
        const timeWindow = this.currentTimeWindow;
        
        // 从全局数据中获取最新的概率数据
        const predictionData = dataManager.data?.predictions[gridData.id];
        if (!predictionData || !predictionData.probabilities) {
            console.error(`No prediction data for grid ${gridData.id}`);
            return;
        }
        
        const probs = predictionData.probabilities[`${timeWindow}days`];

        // 更新标题
        document.getElementById('modalTitle').textContent = 
            `グリッド ${gridData.id} の詳細`;

        // 更新坐标范围
        document.getElementById('coordRange').textContent = 
            `北緯 ${gridData.lat[0]}°-　${gridData.lat[1]}°, 東経 ${gridData.lon[0]}°-　${gridData.lon[1]}°`;

        // 更新地区
        document.getElementById('regions').textContent = 
            gridData.region

        // // 更新都道府县
        // document.getElementById('prefectures').textContent = 
        //     gridData.prefectures.join('、');

        // 更新风险等级 - 使用当前时间窗口的风险等级
        const riskElement = document.getElementById('riskLevel');
        const currentRiskLevel = predictionData.riskLevels ? 
            predictionData.riskLevels[`${timeWindow}days`] : 
            this.calculateRiskLevel(probs.combined);
        
        riskElement.textContent = this.getRiskLevelText(currentRiskLevel);
        riskElement.className = `risk-badge risk-${currentRiskLevel}`;

        // 更新概率条
        this.updateProbabilityBar('M3', probs['M3_4.5']);
        this.updateProbabilityBar('M4', probs['M4.5_5.5']);
        this.updateProbabilityBar('M5', probs['M5.5_6.5']);
        this.updateProbabilityBar('M6', probs['M6.5+']);

        // 显示模态框
        modal.classList.add('show');
    }
    // 更新概率条
    updateProbabilityBar(magnitudeId, probability) {
        const fillElement = document.getElementById(`prob${magnitudeId}`);
        const valueElement = document.getElementById(`prob${magnitudeId}Value`);
        
        fillElement.style.width = `${probability * 100}%`;
        valueElement.textContent = `${(probability * 100).toFixed(1)}%`;

        // 根据概率值调整颜色
        if (probability < 0.1) {
            fillElement.style.background = '#4CAF50';
        } else if (probability < 0.3) {
            fillElement.style.background = '#FFC107';
        } else {
            fillElement.style.background = '#FF9800';
        }
    }

    // 获取风险等级文本
    getRiskLevelText(level) {
        const levelMap = {
            'low': '低',
            'medium': '中',
            'high': '高',
            'extreme': '極高'
        };
        return levelMap[level] || '不明';
    }
}

// 创建全局地图管理器实例
const mapManager = new MapManager();