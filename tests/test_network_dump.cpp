#include "InferenceEngine.hpp"
#include "LumenArena.hpp"
#include "Processor.hpp"
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace lumen;

int main() {
  string model_path = "../models/resnet18-v1-7.onnx";
  string labels_path = "../models/labels.txt";
  string dump_path = "../tests/images/dump_from_network.bin";

  Arena arena(1024 * 1024 * 20);
  InferenceEngine engine(model_path);
  SqueezeNetPreProcessor pre;
  SqueezeNetPostProcessor post(labels_path);

  // 2. Load the Raw Binary Dump
  ifstream file(dump_path, ios::binary | ios::ate);
  if (!file.is_open()) {
    cerr << "[-] Error: Could not open " << dump_path
         << ". Run the server and client first!" << endl;
    return -1;
  }
  streamsize size = file.tellg();
  file.seekg(0, ios::beg);

  // 3. Allocate and Load directly into Arena
  unsigned char *arena_ptr = arena.alloc<unsigned char>(size);
  if (!file.read((char *)arena_ptr, size)) {
    cerr << "[-] Error: Failed to read binary data" << endl;
    return -1;
  }
  cout << "[*] Loaded " << size << " bytes from network dump." << endl;

  // 4. Run Inference
  // If this fails, but 'test_inference' (the one using cat.jpg) passes,
  // then the TCPServer is not receiving the bytes correctly.
  string result = engine.infer(arena, pre, post);

  cout << "\n================================" << endl;
  cout << "DUMP ANALYSIS RESULT: " << result << endl;
  cout << "================================\n" << endl;
  return 0;
}
