package com.earthquake.prediction.model;

import java.util.Map;

public class GridPrediction {
    private String gridId;
    private Map<String, ProbabilityData> probabilities;
    private Map<String, String> riskLevels;
    
    public GridPrediction() {}
    
    public GridPrediction(String gridId) {
        this.gridId = gridId;
    }
    
    // Getters and Setters
    public String getGridId() {
        return gridId;
    }
    
    public void setGridId(String gridId) {
        this.gridId = gridId;
    }
    
    public Map<String, ProbabilityData> getProbabilities() {
        return probabilities;
    }
    
    public void setProbabilities(Map<String, ProbabilityData> probabilities) {
        this.probabilities = probabilities;
    }
    
    public Map<String, String> getRiskLevels() {
        return riskLevels;
    }
    
    public void setRiskLevels(Map<String, String> riskLevels) {
        this.riskLevels = riskLevels;
    }
}