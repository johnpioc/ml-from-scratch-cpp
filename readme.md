# John's Machine Learning Library

Implementing machine learning models from scratch in an effort to learn more about C++, high 
performance programming, software design, and machine learning.

### Models Implemented

- Ordinary Least Squares Linear Regression `OLS()`

### 🧠 Motivation

I built this project in an effort to learn more about C++ and aspects such as high performance programming, software design and common language practices, along with learning more about machine.

I attribute my learning to the following textbooks:

- **Introduction to Statistical Learning** by James, Witten, Hastie and Tibshirani
- **C++ High Performance**: by Andrist and Sehr
- **Beautiful C++** by Davidson and Gregory
- **C++ Software Design** by Iglberger

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
