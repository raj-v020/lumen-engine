#include <iostream>
#include <vector>
#include "Arena.hpp"
#include "TCPServer.hpp"
#include "InferenceEngine.hpp"
#include "Processor.hpp"

using namespace lumen;

int main() {
    // 1. Initialize System Components
    Arena arena(1024 * 1024 * 20); // 20MB for safety

    // 2. Start the Server on Port 8080
    TCPServer server("8080", arena);
    std::cout << "[Lumen] Server listening on port 8080..." << std::endl;

    while (true) {
        try {
            server.run();
        } catch (const std::exception& e) {
            std::cerr << "[Lumen Error] " << e.what() << std::endl;
            arena.reset(); 
        }
    }

    return 0;
}
