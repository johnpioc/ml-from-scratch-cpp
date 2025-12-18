# Machine Learning From Scratch (C++)

Implementations of several Machine Learning models from scratch in C++ with runtime & model accuracy comparisons to equivalent implementations in Python's Tensorflow and Statsmodels.

### 🧠 Motivation

I built this project in an effort to learn more about high-performance C++, C++ Industry practices and machine learning. I attribute my learning to the following textbooks:

- **Introduction to Statistical Learning** by James, Witten, Hastie and Tibshirani
- **C++ High Performance** by Andrist and Sehr
- **Beautiful C++** by Davidson and Gregory
- **C++ Software Design** by Iglberger

### 🚀 Models Implemented
Below is a table comparing the models I've implemented versus the equivalent implementations in Tensorflow and Statsmodels in terms of prediction accuracy and training runtime.

**System Specs:** Mac Mini M4 (2024), 16GB

| Model | Accuracy Metric | Accuracy Comparison to Tensorflow / Statsmodels | Runtime Comparison |
|-|-|-|-|
| Simple Linear Regression | $R^2$ | Achieves 100% to Statsmodel's OLS Algorithm | 1905% Faster than Statsmodels |