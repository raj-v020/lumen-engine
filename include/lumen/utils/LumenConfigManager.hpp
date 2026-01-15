#pragma once

#include <string>
#include <map>
#include <nlohmann/json.hpp>

namespace lumen {
namespace utils {

enum class QueueType     { NAIVE_MUTEX, BATCHED_MUTEX, LOCK_FREE_RING };
enum class AllocatorType { STANDARD, LUMEN_ARENA };
enum class ProcessorType { IMAGENET, RAW_TENSOR, OBJECT_DETECTION };

class ConfigManager {
private:
  ConfigManager() = default;

  QueueType m_queue_type = QueueType::NAIVE_MUTEX;
  AllocatorType m_alloc_type = AllocatorType::STANDARD;
  ProcessorType m_proc_type = ProcessorType::IMAGENET;
  std::string m_telemetry_path = "../results/lumen_baseline.csv";

  std::string m_model_path = "../models/squeezenet1.1-7.onnx"; 
  nlohmann::json m_metadata = {{"label_path", "../models/labels.txt"}};

public:
  static ConfigManager& get() {
    static ConfigManager instance;
    return instance;
  }

  ConfigManager(const ConfigManager&) = delete;
  ConfigManager& operator=(const ConfigManager&) = delete;

  QueueType get_queue_type() const { return m_queue_type; }
  AllocatorType get_alloc_type() const { return m_alloc_type; }
  ProcessorType get_processor_type() const { return m_proc_type; }
  std::string get_telemetry_csv_path() const { return m_telemetry_path; }

  std::string get_model_path() const { return m_model_path; }

  std::string get_metadata_extra(const std::string& key) const {
    if (m_metadata.contains(key)) {
      return m_metadata[key].get<std::string>();
    }
    return "";
  }

  void set_queue_type(QueueType type) { m_queue_type = type; }
  void set_alloc_type(AllocatorType type) { m_alloc_type = type; }
  void set_telemetry_path(const std::string& path) { m_telemetry_path = path; }
  void set_model_path(const std::string& path) { m_model_path = path; }

  void load_from_file(const std::string& path);
  void save_template(const std::string& path);
};

}
}
