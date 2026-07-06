# John's Machine Learning Library

A C++20 library designed for low-latency statistical modelling and machine learning. Visit the [Wiki](https://github.com/johnpioc/ml-from-scratch-cpp/wiki) for documentation on how to use the library.

### 🧠 Motivation

I built this project in an effort to learn more about C++ and aspects such as high performance programming, software design and common language practices, along with learning more about machine.

I attribute my learning to the following textbooks:

- **Introduction to Statistical Learning** by James, Witten, Hastie and Tibshirani
- **C++ High Performance**: by Andrist and Sehr
- **Beautiful C++** by Davidson and Gregory
- **C++ Software Design** by Iglberger

### Running Tests

```shell
cmake -S . -B build
cmake --build build
ctest --test-dir build
```
