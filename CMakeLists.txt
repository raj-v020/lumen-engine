cmake_minimum_required(VERSION 3.10)
project(LumenEngine) 

set(CMAKE_CXX_STANDARD 17)

# 1. Setup ONNX variables
set(ONNX_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib")
set(ONNX_LIB "${ONNX_LIB_DIR}/libonnxruntime.so")
set(ONNX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include")

# 2. Define Source Groups
file(GLOB SOURCES "src/*.cpp")

# Create a list of sources for the library/test (everything except main.cpp)
set(LIB_SOURCES ${SOURCES})
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

# 3. Build Main Server (lumen_engine)
add_executable(lumen_engine ${SOURCES})

# Properties for lumen_engine (MUST come after add_executable)
set_target_properties(lumen_engine PROPERTIES 
    INSTALL_RPATH "${ONNX_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

target_include_directories(lumen_engine PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/lumen"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/utils"
)

target_include_directories(lumen_engine PRIVATE "${ONNX_INCLUDE_DIR}")
target_link_libraries(lumen_engine PRIVATE "${ONNX_LIB}")

# 4. Build Inference Test (inference_test)
find_package(OpenCV REQUIRED)

add_executable(inference_test tests/test_inference.cpp ${LIB_SOURCES})

# Properties for inference_test (MUST come after add_executable)
target_link_libraries(inference_test PRIVATE "${ONNX_LIB}" ${OpenCV_LIBS})

target_include_directories(inference_test PRIVATE 
    "${ONNX_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/lumen"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/utils"
)

set_target_properties(inference_test PROPERTIES 
    INSTALL_RPATH "${ONNX_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# 5. Build Network Dump Test (test_dump)
add_executable(test_dump tests/test_network_dump.cpp ${LIB_SOURCES})
target_link_libraries(test_dump PRIVATE "${ONNX_LIB}")
target_include_directories(test_dump PRIVATE 
    "${ONNX_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/lumen"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/utils"
)
