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

target_link_libraries(jmll_bench_deps PRIVATE jmll::jmll)

add_executable(ols_bench ols_bench.cpp)

target_link_libraries(ols_bench PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    jmll::jmll
    jmll_bench_deps
)

