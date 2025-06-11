package com.earthquake.prediction.service;

import java.io.IOException;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.fasterxml.jackson.databind.ObjectMapper;

import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;

@Service
public class S3Service {
    
    private final S3Client s3Client;
    private final ObjectMapper objectMapper;
    
    @Value("${aws.s3.bucket}")
    private String bucketName;
    
    @Value("${aws.s3.predictions.key}")
    private String predictionsKey;
    
    @Value("${aws.region}")
    private String awsRegion;
    
    public S3Service() {
        this.s3Client = S3Client.builder()
                .region(Region.of("ap-northeast-1"))
                .build();
        this.objectMapper = new ObjectMapper();
    }
    
    /**
     * 从S3读取预测数据
     * 注：实际部署时取消注释
     */
    public String readPredictionsFromS3() throws IOException {
        /*
        // 构建请求
        GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                .bucket(bucketName)
                .key(predictionsKey)
                .build();
        
        // 读取数据
        byte[] data = s3Client.getObject(getObjectRequest).readAllBytes();
        return new String(data);
        */
        
        // 暂时返回null，使用模拟数据
        return null;
    }
}