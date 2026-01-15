#include <lumen/utils/LumenConfigManager.hpp>
#include <fstream>
#include <iostream>

namespace lumen {
namespace utils {

using json = nlohmann::json;

void ConfigManager::load_from_file(const std::string& path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        std::cout << "[CONFIG] " << path << " not found. Using baseline defaults." << std::endl;
        return;
    }

    try {
        json j;
        file >> j;

        if (j.contains("engine")) {
            auto& eng = j["engine"];
            
            std::string q_type = eng.value("queue_type", "naive_mutex");
            if (q_type == "batched_mutex") m_queue_type = QueueType::BATCHED_MUTEX;
            else if (q_type == "lock_free") m_queue_type = QueueType::LOCK_FREE_RING;
            else m_queue_type = QueueType::NAIVE_MUTEX;

            std::string a_type = eng.value("allocator_type", "standard");
            if (a_type == "lumen_arena") m_alloc_type = AllocatorType::LUMEN_ARENA;
            else m_alloc_type = AllocatorType::STANDARD;

            m_telemetry_path = eng.value("telemetry_csv_path", "lumen_baseline.csv");
        }

        if (j.contains("model")) {
            auto& mod = j["model"];
            m_model_path = mod.value("path", "");
            
            std::string p_type = mod.value("processor_type", "imagenet");
            if (p_type == "raw") m_proc_type = ProcessorType::RAW_TENSOR;
            else if (p_type == "detection") m_proc_type = ProcessorType::OBJECT_DETECTION;
            else m_proc_type = ProcessorType::IMAGENET;

            m_model_path = mod.value("path", "");

            if (mod.contains("metadata")) {
                m_metadata = mod["metadata"];
            }
        }

        std::cout << "[CONFIG] Successfully loaded: " << path << std::endl;
        std::cout << "         Queue: " << (int)m_queue_type << " | Alloc: " << (int)m_alloc_type << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[CONFIG] Error parsing JSON: " << e.what() << ". Falling back to defaults." << std::endl;
    }
}
void ConfigManager::save_template(const std::string& path) {
    json j = {
        {"engine", {
            {"queue_type", "naive_mutex"},
            {"allocator_type", "standard"},
            {"telemetry_csv_path", "lumen_baseline.csv"}
        }},
        {"model", {
            {"path", "models/squeezenet1.1.onnx"},
            {"processor_type", "imagenet"}, 
            {"metadata", {
                {"type", "classification"},
                {"label_path", "models/synset_words.txt"}
            }}
        }}
    };
    
    std::ofstream file(path);
    if (file.is_open()) {
        file << j.dump(4);
        std::cout << "[CONFIG] Created default template config at: " << path << std::endl;
    }
}

}
}
