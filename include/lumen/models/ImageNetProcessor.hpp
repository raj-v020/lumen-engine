#pragma once

#include <lumen/interfaces/IProcessor.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>

namespace lumen {
namespace core {

class ImageNetPreProcessor : public interfaces::IPreProcessor {
public:
    void transform(const uint8_t* pixels, float* tensor_buffer, size_t width, size_t height) override {
        
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std[3]  = {0.229f, 0.224f, 0.225f};
        
        size_t area = width * height;

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                for (size_t c = 0; c < 3; ++c) {
                    // Source (HWC): The raw bytes usually come in as RGBRGB...
                    size_t src_idx = (y * width + x) * 3 + c;

                    // Destination (NCHW): Planar format RRR...GGG...BBB...
                    size_t dst_idx = c * area + (y * width + x);

                    float val = static_cast<float>(pixels[src_idx]);
                    tensor_buffer[dst_idx] = ((val / 255.0f) - mean[c]) / std[c];
                }
            }
        }
    }
};

class ImageNetPostProcessor : public interfaces::IPostProcessor {
private:
    std::vector<std::string> labels;

public:
    explicit ImageNetPostProcessor(const std::string& label_path) {
        std::ifstream file(label_path);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) labels.push_back(line);
            }
        } else {
            std::cerr << "[Lumen Warning] Labels file not found: " << label_path << std::endl;
        }
    }

    std::string handle_results(const float* results_ptr, size_t count) override {
        if (!results_ptr || count == 0) return "Error: Empty or Invalid Tensor";

        const float* it = std::max_element(results_ptr, results_ptr + count);
        int class_id = static_cast<int>(it - results_ptr);
        float max_val = *it;

        float sum_exp = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            sum_exp += std::exp(results_ptr[i] - max_val);
        }

        float confidence = 1.0f / sum_exp;

        std::string label_name = (class_id < (int)labels.size()) ? labels[class_id] : "Unknown";

        std::cout << "\033[1;32m[RESULT]\033[0m Class: " << label_name 
            << " (" << (int)(confidence * 100) << "%)" << std::endl;

        return label_name;
    }
};

// Aliases to ensure main.cpp works without changes
using SqueezeNetPreProcessor = ImageNetPreProcessor;
using SqueezeNetPostProcessor = ImageNetPostProcessor;

using ResNetPreProcessor = ImageNetPreProcessor;
using ResNetPostProcessor = ImageNetPostProcessor;

}
}
