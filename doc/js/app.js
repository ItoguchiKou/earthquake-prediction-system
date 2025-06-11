// 主应用程序
class EarthquakeApp {
    constructor() {
        this.currentTimeWindow = 7;
        this.isLoading = false;
    }

    // 初始化应用
    async init() {
        try {
            // 显示加载器
            this.showLoader();

            // 初始化地图
            mapManager.initMap();

            // 加载数据
            const data = await dataManager.fetchPredictions();
            if (!data) {
                throw new Error('Failed to load prediction data');
            }

            // 显示初始数据
            this.updateDisplay(this.currentTimeWindow);

            // 绑定事件
            this.bindEvents();

            // 隐藏加载器
            this.hideLoader();

        } catch (error) {
            console.error('App initialization failed:', error);
            this.hideLoader();
            alert('データの読み込みに失敗しました。ページを更新してください。');
        }
    }

    // 绑定事件
    bindEvents() {
        // 时间窗口切换
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const timeWindow = parseInt(e.target.dataset.window);
                this.switchTimeWindow(timeWindow);
            });
        });

        // 模态框关闭按钮
        document.querySelector('.close-btn').addEventListener('click', () => {
            this.closeModal();
        });

        // 点击模态框外部关闭
        document.getElementById('gridModal').addEventListener('click', (e) => {
            if (e.target.id === 'gridModal') {
                this.closeModal();
            }
        });

        // 防止模态框内容区域的点击事件冒泡
        document.querySelector('.modal-content').addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }

    // 切换时间窗口
    switchTimeWindow(timeWindow) {
        if (this.currentTimeWindow === timeWindow) return;

        // 更新按钮状态
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-window="${timeWindow}"]`).classList.add('active');

        // 更新当前时间窗口
        this.currentTimeWindow = timeWindow;
        dataManager.currentTimeWindow = timeWindow;

        // 更新显示
        this.updateDisplay(timeWindow);
    }

    // 更新显示
    updateDisplay(timeWindow) {
        // 获取当前时间窗口的数据
        const gridData = dataManager.getDataForTimeWindow(timeWindow);
        
        if (!gridData) {
            console.error('No data available for time window:', timeWindow);
            return;
        }

        // 更新地图网格
        mapManager.updateGridColors(timeWindow);

        // 更新底部信息栏
        this.updateInfoBar();
    }

    // 更新信息栏
    updateInfoBar() {
        // 更新时间
        const updateTime = dataManager.formatUpdateTime(dataManager.data.updateTime);
        document.getElementById('updateTime').textContent = updateTime;

        // 更新最高风险
        const maxRisk = dataManager.getMaxRisk();
        document.getElementById('maxRisk').textContent = `${(maxRisk * 100).toFixed(1)}%`;
    }

    // 显示加载器
    showLoader() {
        document.getElementById('loader').classList.remove('hide');
        this.isLoading = true;
    }

    // 隐藏加载器
    hideLoader() {
        document.getElementById('loader').classList.add('hide');
        this.isLoading = false;
    }

    // 关闭模态框
    closeModal() {
        document.getElementById('gridModal').classList.remove('show');
    }
}

// 应用启动
document.addEventListener('DOMContentLoaded', async () => {
    const app = new EarthquakeApp();
    await app.init();

    // 初始加载网格数据
    const initialData = dataManager.getDataForTimeWindow(7);
    if (initialData) {
        mapManager.updateGrids(initialData);
    }
});

// 处理页面可见性变化（从后台切换回来时）
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        // 可以在这里添加数据刷新逻辑
        console.log('Page became visible');
    }
});

// 处理网络状态变化
window.addEventListener('online', () => {
    console.log('Network connection restored');
    // 可以在这里尝试重新加载数据
});

window.addEventListener('offline', () => {
    console.log('Network connection lost');
    // 可以在这里显示离线提示
});