# John's Machine Learning Library

Implementing machine learning models from scratch in an effort to learn more about C++, high 
performance programming, software design, and machine learning.

### Models Implemented

- Ordinary Least Squares Linear Regression `OLS()`

### Running Benchmarks

```shell
# Make sure mlpack dependencies are installed
brew install ensmallen armadillo cereal

cmake -S . -B build
cmake --build build
./build/benchmarks/ols_bench
```

### Running Tests

```shell
# Make sure mlpack dependencies are installed
brew install ensmallen armadillo cereal

cmake -S . -B build
cmake --build build
ctest --test-dir build
```
