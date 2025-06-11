package com.earthquake.prediction.service;

import com.earthquake.prediction.model.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import java.time.Instant;
import java.util.*;

@Service
public class PredictionService {
    
    @Autowired
    private S3Service s3Service;
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    public PredictionResponse getPredictions() {
        try {
            // 尝试从S3读取数据
            String s3Data = s3Service.readPredictionsFromS3();
            if (s3Data != null) {
                // 处理S3数据并返回
                return processS3Data(s3Data);
            }
        } catch (Exception e) {
            // 记录错误，使用模拟数据
            System.err.println("Failed to read from S3: " + e.getMessage());
        }
        
        // 使用模拟数据
        return generateMockData();
    }
    
    /**
     * 处理S3数据（16个网格的数据）
     * 预期的S3数据格式：
     * {
     *   "predictions": [
     *     [grid0的12个值], 
     *     [grid1的12个值],
     *     ...
     *     [grid15的12个值]
     *   ],
     *   "updateTime": "2024-01-15T00:00:00Z"
     * }
     * 
     * 每个网格的12个值按顺序为：
     * [7天M3-4.5, 7天M4.5-5.5, 7天M5.5-6.5, 7天M6.5+,
     *  14天M3-4.5, 14天M4.5-5.5, 14天M5.5-6.5, 14天M6.5+,
     *  30天M3-4.5, 30天M4.5-5.5, 30天M5.5-6.5, 30天M6.5+]
     */
    private PredictionResponse processS3Data(String s3Data) {
        try {
            // 解析S3数据
            Map<String, Object> s3Response = objectMapper.readValue(s3Data, 
                new TypeReference<Map<String, Object>>() {});
            
            // 获取预测数据数组
            List<List<Double>> predictionsArray = (List<List<Double>>) s3Response.get("predictions");
            String updateTime = (String) s3Response.getOrDefault("updateTime", Instant.now().toString());
            
            Map<String, GridPrediction> predictions = new HashMap<>();
            
            // 处理16个网格的数据
            for (int gridIndex = 0; gridIndex < 16; gridIndex++) {
                String gridId = String.valueOf(gridIndex + 1); // 网格ID从1开始
                List<Double> gridData = predictionsArray.get(gridIndex);
                
                GridPrediction gridPrediction = new GridPrediction(gridId);
                Map<String, ProbabilityData> probabilities = new HashMap<>();
                Map<String, String> riskLevels = new HashMap<>();
                
                // 解析12个预测值
                int dataIndex = 0;
                for (int days : Arrays.asList(7, 14, 30)) {
                    // 读取4个震级的概率值
                    double m3_45 = gridData.get(dataIndex++);
                    double m45_55 = gridData.get(dataIndex++);
                    double m55_65 = gridData.get(dataIndex++);
                    double m65_plus = gridData.get(dataIndex++);
                    
                    // 创建概率数据对象
                    ProbabilityData probData = new ProbabilityData(m3_45, m45_55, m55_65, m65_plus);
                    probabilities.put(days + "days", probData);
                    
                    // 计算风险等级
                    riskLevels.put(days + "days", getRiskLevel(probData.getCombined()));
                }
                
                gridPrediction.setProbabilities(probabilities);
                gridPrediction.setRiskLevels(riskLevels);
                predictions.put(gridId, gridPrediction);
            }
            
            return new PredictionResponse(updateTime, predictions);
            
        } catch (Exception e) {
            System.err.println("Error processing S3 data: " + e.getMessage());
            e.printStackTrace();
            // 如果处理失败，返回模拟数据
            return generateMockData();
        }
    }
    
    /**
     * 生成模拟数据（16个网格）
     */
    private PredictionResponse generateMockData() {
        Map<String, GridPrediction> predictions = new HashMap<>();
        
        // 为16个网格生成预测数据
        for (int gridId = 1; gridId <= 16; gridId++) {
            String gridIdStr = String.valueOf(gridId);
            GridPrediction gridPrediction = new GridPrediction(gridIdStr);
            
            // 根据网格位置设置基础风险
            double baseRisk = getBaseRisk(gridIdStr);
            
            // 生成三个时间窗口的预测
            Map<String, ProbabilityData> probabilities = new HashMap<>();
            Map<String, String> riskLevels = new HashMap<>();
            
            for (int days : Arrays.asList(7, 14, 30)) {
                double timeMultiplier = Math.sqrt(days / 7.0) * 0.5;
                
                // 生成四个震级的概率
                double m3_45 = generateProbability(baseRisk * 0.8 + timeMultiplier * 0.05);
                double m45_55 = generateProbability(baseRisk * 0.3 + timeMultiplier * 0.02);
                double m55_65 = generateProbability(baseRisk * 0.1 + timeMultiplier * 0.01);
                double m65_plus = generateProbability(baseRisk * 0.02 + timeMultiplier * 0.002);
                
                ProbabilityData probData = new ProbabilityData(m3_45, m45_55, m55_65, m65_plus);
                probabilities.put(days + "days", probData);
                
                // 计算风险等级
                riskLevels.put(days + "days", getRiskLevel(probData.getCombined()));
            }
            
            gridPrediction.setProbabilities(probabilities);
            gridPrediction.setRiskLevels(riskLevels);
            predictions.put(gridIdStr, gridPrediction);
        }
        
        return new PredictionResponse(Instant.now().toString(), predictions);
    }
    
    /**
     * 根据网格ID获取基础风险值
     */
    private double getBaseRisk(String gridId) {
        switch (gridId) {
            case "11": return 0.15;  // 関東北東部（高风险）
            case "13": return 0.12;  // 東北太平洋側（高风险）
            case "8":  return 0.10;  // 関東南部（中高风险）
            case "10": return 0.08;  // 中部・関東北部（中风险）
            case "15": return 0.07;  // 北海道中部（中风险）
            case "16": return 0.07;  // 北海道東部（中风险）
            case "7":  return 0.06;  // 近畿・四国東部（中低风险）
            case "12": return 0.06;  // 東北日本海側（中低风险）
            case "14": return 0.05;  // 北海道西部（低风险）
            default:   return 0.05;  // その他（低风险）
        }
    }
    
    /**
     * 生成概率值（0-1之间）
     */
    private double generateProbability(double baseProb) {
        double variance = 0.1;
        double prob = baseProb + (Math.random() - 0.5) * variance;
        return Math.max(0, Math.min(1, prob));
    }
    
    /**
     * 根据概率计算风险等级
     */
    private String getRiskLevel(double probability) {
        if (probability < 0.1) return "low";
        if (probability < 0.3) return "medium";
        if (probability < 0.5) return "high";
        return "extreme";
    }
}