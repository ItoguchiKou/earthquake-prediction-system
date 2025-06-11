package com.earthquake.prediction.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.earthquake.prediction.model.PredictionResponse;
import com.earthquake.prediction.service.PredictionService;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class PredictionController {
    
    @Autowired
    private PredictionService predictionService;
    
    @GetMapping("/predictions")
    public PredictionResponse getPredictions() {
        return predictionService.getPredictions();
    }
    
    @GetMapping("/health")
    public String health() {
        return "OK";
    }
}