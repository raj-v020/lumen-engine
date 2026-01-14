#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include "IProcessor.hpp"

namespace lumen {
class ImageNetPreProcessor : public IPreProcessor {
public:
    /**
     * ImageNet Requirements:
     * 1. Rescale pixels from [0, 255] to [0, 1]
     * 2. Normalize using ImageNet Mean: [0.485, 0.456, 0.406]
     * 3. Normalize using ImageNet Std:  [0.229, 0.224, 0.225]
     * 4. Format: NCHW (Channels first)
     */
    void transform(void* raw_input, float* tensor_buffer, size_t width, size_t height) override {
        uint8_t* pixels = static_cast<uint8_t*>(raw_input);
        
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std[3] = {0.229f, 0.224f, 0.225f};
        size_t area = width * height;

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                for (size_t c = 0; c < 3; ++c) {
                    size_t src_idx = (y * width + x) * 3 + c;
                    size_t dst_idx = c * area + (y * width + x);
                    
                    float val = static_cast<float>(pixels[src_idx]);
                    tensor_buffer[dst_idx] = ((val / 255.0f) - mean[c]) / std[c];
                }
            }
        }
    }
};

class ImageNetPostProcessor : public IPostProcessor {
private:
    std::vector<std::string> labels;

public:
    ImageNetPostProcessor(const std::string& label_path) {
        std::ifstream file(label_path);
        std::string line;
        while (std::getline(file, line)) {
            labels.push_back(line);
        }
        if (labels.empty()) {
            std::cerr << "[Warning] Labels file empty or not found at: " << label_path << std::endl;
        }
    }

    std::string handle_results(const std::vector<float>& results) override {
        // 1. Find the Argmax
        auto it = std::max_element(results.begin(), results.end());
        int class_id = std::distance(results.begin(), it);

        // 2. Apply Softmax for real percentage
        float sum_exp = 0.0f;
        for (float val : results) {
            sum_exp += std::exp(val - *it); 
        }
        float confidence = std::exp(*it - *it) / sum_exp;

        std::string label_name = (class_id < labels.size()) ? labels[class_id] : "Unknown";

        std::cout << "\n------------------------------------" << std::endl;
        std::cout << " LUMEN INFERENCE RESULT " << std::endl;
        std::cout << " Identified: " << label_name << " (ID: " << class_id << ")" << std::endl;
        std::cout << " Confidence: " << (confidence * 100.0f) << "%" << std::endl;
        std::cout << "------------------------------------\n" << std::endl;
        return label_name + " (" + std::to_string(confidence) + ")";
    }
};
}

