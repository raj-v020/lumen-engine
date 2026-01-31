#include "InferenceEngine.hpp"
#include "LumenArena.hpp"
#include "Processor.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace lumen;

int main() {
  string model_path = "../models/resnet18-v1-7.onnx";
  string labels_path = "../models/labels.txt";
  string image_path = "../tests/images/cat.jpg";

  LumenArena arena(1024 * 1024 * 20);
  InferenceEngine engine(model_path);
  SqueezeNetPreProcessor pre;
  SqueezeNetPostProcessor post(labels_path);

  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    cerr << "Error: Could not load image at " << image_path << endl;
    return -1;
  }

  cv::resize(img, img, cv::Size(224, 224));
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

  size_t image_size = img.total() * img.elemSize(); // 224 * 224 * 3
  unsigned char *arena_ptr = arena.alloc<unsigned char>(image_size);

  std::memcpy(arena_ptr, img.data, image_size);

  cout << "[Test] Data loaded into Arena at: " << (void *)arena_ptr << endl;

  string result = engine.infer(arena, pre, post);

  cout << "-------------------------------" << endl;
  cout << "RESULT: " << result << endl;
  cout << "-------------------------------" << endl;

  return 0;
}
