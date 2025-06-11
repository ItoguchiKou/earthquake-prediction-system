package com.earthquake.prediction.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ProbabilityData {
    @JsonProperty("M3_4.5")
    private double m3_45;
    
    @JsonProperty("M4.5_5.5")
    private double m45_55;
    
    @JsonProperty("M5.5_6.5")
    private double m55_65;
    
    @JsonProperty("M6.5+")
    private double m65_plus;
    
    private double combined;
    
    public ProbabilityData() {}
    
    public ProbabilityData(double m3_45, double m45_55, double m55_65, double m65_plus) {
        this.m3_45 = m3_45;
        this.m45_55 = m45_55;
        this.m55_65 = m55_65;
        this.m65_plus = m65_plus;
        this.combined = Math.max(Math.max(m3_45, m45_55), Math.max(m55_65, m65_plus));
    }
    
    // Getters and Setters
    public double getM3_45() {
        return m3_45;
    }
    
    public void setM3_45(double m3_45) {
        this.m3_45 = m3_45;
    }
    
    public double getM45_55() {
        return m45_55;
    }
    
    public void setM45_55(double m45_55) {
        this.m45_55 = m45_55;
    }
    
    public double getM55_65() {
        return m55_65;
    }
    
    public void setM55_65(double m55_65) {
        this.m55_65 = m55_65;
    }
    
    public double getM65_plus() {
        return m65_plus;
    }
    
    public void setM65_plus(double m65_plus) {
        this.m65_plus = m65_plus;
    }
    
    public double getCombined() {
        return combined;
    }
    
    public void setCombined(double combined) {
        this.combined = combined;
    }
}