include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)

FetchContent_MakeAvailable(googlebenchmark)

# Benchmark dependencies
add_library(jmll_bench_deps STATIC
    data_generation.cpp
)

# mlpack dependencies
find_package(Armadillo REQUIRED)

if (NOT TARGET Armadillo::Armadillo)
    add_library(Armadillo::Armadillo INTERFACE IMPORTED)
    set_target_properties(Armadillo::Armadillo PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ARMADILLO_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${ARMADILLO_LIBRARIES}"
    )
endif()

set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
find_package(OpenMP REQUIRED)

set(cereal_DIR "/opt/homebrew/opt/cereal/lib/cmake/cereal" CACHE PATH "Path to cereal")
find_package(cereal REQUIRED)

set(ensmallen_DIR "/opt/homebrew/opt/ensmallen/lib/cmake/ensmallen" CACHE PATH "Path to ensmallen")
find_package(ensmallen REQUIRED)
# mlpack
add_library(mlpack INTERFACE)

target_include_directories(mlpack INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/mlpack/src
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(mlpack INTERFACE
    ${ARMADILLO_LIBRARIES}
    cereal::cereal
    ensmallen::ensmallen
)

target_link_libraries(jmll_bench_deps PRIVATE jmll::jmll)

add_executable(ols_bench ols_bench.cpp)

target_compile_definitions(ols_bench PRIVATE ARMA_WARN_LEVEL=0)

target_link_libraries(ols_bench PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    jmll::jmll
    jmll_bench_deps
    mlpack
)


