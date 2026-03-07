cmake_minimum_required(VERSION 3.20)
project(jmll)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(jmll
    src/main.cpp
    src/core/matrix.cpp
    src/core/vector.cpp
    src/models/tuning/evaluation_metric.cpp
    src/models/tuning/fitting_policy.cpp
    src/data_preprocessing/datasets.cpp
    src/data_preprocessing/train_test_split.cpp
)

target_include_directories(jmll PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
