cmake_minimum_required(VERSION 3.10)
project(LumenEngine) 

set(CMAKE_CXX_STANDARD 17)


set(ONNX_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib")
set(ONNX_LIB "${ONNX_LIB_DIR}/libonnxruntime.so")

add_executable(lumen_engine 
    src/main.cpp
    src/Arena.cpp
    src/InferenceEngine.cpp
    src/TCPServer.cpp)

target_link_libraries(lumen_engine PRIVATE "${ONNX_LIB}")
target_include_directories(lumen_engine PRIVATE 
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include"
)
target_include_directories(lumen_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/lumen")
target_include_directories(lumen_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/utils")

set_target_properties(lumen_engine PROPERTIES 
    INSTALL_RPATH "${ONNX_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)
