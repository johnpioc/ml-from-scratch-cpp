cmake_minimum_required(VERSION 3.20)
project(mlfs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mlfs
    src/main.cpp
    src/core/matrix.cpp
    src/core/vector.cpp
    src/models/linear_regression.cpp
)

target_include_directories(mlfs PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
