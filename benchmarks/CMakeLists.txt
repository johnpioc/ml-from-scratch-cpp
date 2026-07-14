include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(benchmark 
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)

# cereal header
set(BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_SANDBOX OFF CACHE BOOL "" FORCE)
set(SKIP_PERFORMANCE_COMPARISON ON CACHE BOOL "" FORCE)
FetchContent_Declare(cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    GIT_TAG v1.3.2
)

# armadillo, ensmallen and mlpack headers
FetchContent_Declare(armadillo
    GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
    GIT_TAG 14.2.x
    SOURCE_SUBDIR no-cmake-please
)
FetchContent_Declare(ensmallen
    GIT_REPOSITORY https://github.com/mlpack/ensmallen.git
    GIT_TAG 2.22.1
    SOURCE_SUBDIR no-cmake-please
)
FetchContent_Declare(mlpack
    GIT_REPOSITORY https://github.com/mlpack/mlpack.git
    GIT_TAG 4.8.0
    SOURCE_SUBDIR no-cmake-please
)

FetchContent_MakeAvailable(benchmark cereal armadillo ensmallen mlpack)

add_library(mlpack_stack INTERFACE)
target_include_directories(mlpack_stack SYSTEM INTERFACE
    ${armadillo_SOURCE_DIR}/include
    ${ensmallen_SOURCE_DIR}/include
    ${mlpack_SOURCE_DIR}/src
)

target_link_libraries(mlpack_stack INTERFACE cereal::cereal)
target_compile_definitions(mlpack_stack INTERFACE
    ARMA_DONT_USE_WRAPPER
    ARMA_USE_BLAS
    ARMA_USE_LAPACK
)

# accelerate on macos, else use openblas
if (APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(mlpack_stack INTERFACE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(mlpack_stack INTERFACE ACCELERATE_NEW_LAPACK)
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    target_link_libraries(mlpack_stack INTERFACE BLAS::BLAS LAPACK::LAPACK)
endif()

add_executable(jmll_bench
    # empty for now
)
target_link_libraries(jmll_bench PRIVATE
    jmll
    mlpack_stack
    benchmark::benchmark_main
)
target_compile_features(jmll_bench PRIVATE cxx_std_23)
