cmake_minimum_required(VERSION 3.10)
project(InferenceEngine) 

set(CMAKE_CXX_STANDARD 17)
add_executable(engine 
    src/main.cpp
    src/LinearAllocator.cpp)
target_include_directories(engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
