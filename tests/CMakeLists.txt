include(FetchContent)

# =================================================================================================
# GOOGLE TEST
# =================================================================================================
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# =================================================================================================
# Eigen
# =================================================================================================
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)
set(BUILD_TESTING ON CACHE BOOL "" FORCE)

include(GoogleTest)
add_executable((matrix_test core/matrix_test.cpp)
target_link_libraries(matrix_test PRIVATE
  jmll::jmll
  GTest::gtest_main
  Eigen3::Eigen
)