cmake_minimum_required(VERSION 3.10)
project(LumenEngine) 

set(CMAKE_CXX_STANDARD 17)

# Step 1: Declare the configuration variable with a default value of "None"
set(LUMEN_ENABLE_SANITIZER "None" CACHE STRING "Compile with runtime sanitizers")

# Step 2: Restrict the valid input strings to an explicit list
set_property(CACHE LUMEN_ENABLE_SANITIZER PROPERTY STRINGS "None" "Address" "Thread" "Undefined")

# 1. Setup ONNX Variables
set(ONNX_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib")
set(ONNX_LIB "${ONNX_LIB_DIR}/libonnxruntime.so")
set(ONNX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include")

# 2. Recursive Source Discovery
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")

# Create a list of sources (everything except main.cpp) for tests
set(LIB_SOURCES ${SOURCES})
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

# 3. Build Main Server (lumen_engine)
add_executable(lumen_engine ${SOURCES})

# Properties for RPATH (So the binary knows where to find libonnxruntime.so at runtime)
set_target_properties(lumen_engine PROPERTIES 
    INSTALL_RPATH "${ONNX_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# --- UPDATED: Include Strategy ---
# 1. Project Headers: include/lumen/...
# 2. Third Party: third_party/nlohmann/...
target_include_directories(lumen_engine PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

target_include_directories(lumen_engine PRIVATE "${ONNX_INCLUDE_DIR}")
target_link_libraries(lumen_engine PRIVATE "${ONNX_LIB}" pthread)

# --- NEW: Automation for Config ---
# This copies your config.json to the build folder automatically
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config.json")
    configure_file(config.json config.json COPYONLY)
endif()

# 4. Build Inference Test (inference_test)
#find_package(OpenCV REQUIRED)

#add_executable(inference_test tests/test_inference.cpp ${LIB_SOURCES})

#target_link_libraries(inference_test PRIVATE "${ONNX_LIB}" ${OpenCV_LIBS} pthread)

#target_include_directories(inference_test PRIVATE 
#    "${CMAKE_CURRENT_SOURCE_DIR}/include"
#    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
#    "${ONNX_INCLUDE_DIR}"
#)

#set_target_properties(inference_test PROPERTIES 
#    INSTALL_RPATH "${ONNX_LIB_DIR}"
#    BUILD_WITH_INSTALL_RPATH TRUE
#)

# 5. Build Network Dump Test (test_dump)
#add_executable(test_dump tests/test_network_dump.cpp ${LIB_SOURCES})
#target_link_libraries(test_dump PRIVATE "${ONNX_LIB}" pthread)
#target_include_directories(test_dump PRIVATE 
#    "${CMAKE_CURRENT_SOURCE_DIR}/include"
#    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
#    "${ONNX_INCLUDE_DIR}"
#)

# 6. Build Fuzz Concurrency Test (fuzz_concurrency)
add_executable(fuzz_concurrency tests/fuzz_concurrency.cpp ${LIB_SOURCES})
target_link_libraries(fuzz_concurrency PRIVATE "${ONNX_LIB}" pthread)
target_include_directories(fuzz_concurrency PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    "${ONNX_INCLUDE_DIR}"
)
set_target_properties(fuzz_concurrency PROPERTIES 
    INSTALL_RPATH "${ONNX_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(NOT LUMEN_ENABLE_SANITIZER STREQUAL "None")
    if(LUMEN_ENABLE_SANITIZER STREQUAL "Address")
        message(STATUS "[BUILD] Configuring compilation for AddressSanitizer...")
        target_compile_options(fuzz_concurrency PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(fuzz_concurrency PRIVATE -fsanitize=address)
    elseif(LUMEN_ENABLE_SANITIZER STREQUAL "Thread")
        message(STATUS "[BUILD] Configuring compilation for ThreadSanitizer...")
        target_compile_options(fuzz_concurrency PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
        target_link_options(fuzz_concurrency PRIVATE -fsanitize=thread)
    endif()
endif()

# ==========================================
# Target: Latency Regression Benchmark Suite
# ==========================================
add_executable(bench_pure_latency tests/bench_pure_latency.cpp ${LIB_SOURCES})

target_link_libraries(bench_pure_latency PRIVATE "${ONNX_LIB}" pthread)

target_include_directories(bench_pure_latency PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    "${ONNX_INCLUDE_DIR}"
)

# Ensure full optimization and native hardware instruction generation for profiling
target_compile_options(bench_pure_latency PRIVATE -O3 -march=native)

# Forward system or CI environment overrides down into the C++ preprocessor symbols
target_compile_definitions(bench_pure_latency PRIVATE ${COMPILE_DEFINITIONS})

set_target_properties(bench_pure_latency PROPERTIES 
    INSTALL_RPATH "${ONNX_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)
