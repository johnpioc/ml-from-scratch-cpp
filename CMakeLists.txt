cmake_minimum_required(VERSION 4.1.2)
project(neural_network_cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(${PROJECT_NAME} src/main.cpp)
add_library(structures src/Structures.cpp)

target_link_libraries(${PROJECT_NAME} structures)
target_include_directories(${PROJECT_NAME} PRIVATE src)
