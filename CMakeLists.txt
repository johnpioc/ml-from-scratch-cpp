cmake_minimum_required(VERSION 4.1.2)
project(ml_from_scratch)

add_executable(${PROJECT_NAME} src/main.cpp)
add_library(matrix src/Matrix.cpp)
add_library(linear_regression src/models/LinearRegression.cpp)

target_link_libraries(${PROJECT_NAME} matrix)
target_link_libraries(${PROJECT_NAME} linear_regression)
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_include_directories(${PROJECT_NAME} PRIVATE src/models)
