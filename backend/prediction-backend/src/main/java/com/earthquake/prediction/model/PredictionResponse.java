package com.earthquake.prediction.model;

import java.util.Map;

public class PredictionResponse {
    private String updateTime;
    private Map<String, GridPrediction> predictions;
    
    public PredictionResponse() {}
    
    public PredictionResponse(String updateTime, Map<String, GridPrediction> predictions) {
        this.updateTime = updateTime;
        this.predictions = predictions;
    }
    
    // Getters and Setters
    public String getUpdateTime() {
        return updateTime;
    }
    
    public void setUpdateTime(String updateTime) {
        this.updateTime = updateTime;
    }
    
    public Map<String, GridPrediction> getPredictions() {
        return predictions;
    }
    
    public void setPredictions(Map<String, GridPrediction> predictions) {
        this.predictions = predictions;
    }
}