cmake_minimum_required(VERSION 3.20)
project(jmll)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE JMLL_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/models/*.cpp
)

add_library(jmll STATIC ${JMLL_SOURCES})

target_include_directories(jmll PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(jmll PUBLIC cxx_std_20)

add_library(jmll::jmll ALIAS jmll)

option(BUILD_TESTS "Build the test suite" ON)
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
