#include <fstream>
#include <iostream>
#include <lumen/core/InferenceEngine.hpp>
#include <lumen/memory/LumenArena.hpp>
#include <lumen/models/ImageNetProcessor.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

int main() {
  string model_path = "../models/resnet18-v1-7.onnx";
  string labels_path = "../models/labels.txt";
  string image_path = "../tests/images/cat.jpg";

  lumen::memory::LumenArena arena(1024 * 1024 * 20);
  lumen::core::InferenceEngine engine(model_path);
  lumen::models::SqueezeNetPreProcessor pre;
  lumen::models::SqueezeNetPostProcessor post(labels_path);

  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    cerr << "Error: Could not load image at " << image_path << endl;
    return -1;
  }

  cv::resize(img, img, cv::Size(224, 224));
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

  size_t image_size = img.total() * img.elemSize(); // 224 * 224 * 3
  unsigned char *arena_ptr = arena.Alloc<unsigned char>(image_size);

  std::memcpy(arena_ptr, img.data, image_size);

  cout << "[Test] Data loaded into Arena at: " << (void *)arena_ptr << endl;

  string result = engine.infer(arena, pre, post);

  cout << "-------------------------------" << endl;
  cout << "RESULT: " << result << endl;
  cout << "-------------------------------" << endl;

  return 0;
}
