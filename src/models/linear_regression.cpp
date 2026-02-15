#include "mlfs/core/fitting_policy.hpp"
#include <cmath>
#include <mlfs/core/matrix.hpp>
#include <mlfs/core/vector.hpp>
#include <mlfs/models/linear_regression.hpp>

using namespace mlfs::models;
using namespace mlfs::core;

template<LinearRegressionFittingPolicy FittingPolicy>
template<typename... Args>
void LinearRegression<FittingPolicy>::fit(Matrix& x, Vector& y, Args... args) {
    this->beta_ = this->policy_.fit(x, y, args...);
}

template<LinearRegressionFittingPolicy FittingPolicy>
Vector LinearRegression<FittingPolicy>::predict(Matrix& x) {
    Matrix augmented = x.prependOnes();
    return augmented * this->beta_;
}

template<LinearRegressionFittingPolicy FittingPolicy>
double LinearRegression<FittingPolicy>::evaluate(Vector& yPred, Vector& yTrue) {
    int n = yPred.numCells;

    // Calculate mean response
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += yTrue.get(i);
    }
    double yHat = sum / n;

    // Calculate tss and rss
    double tss = 0, rss = 0;
    for (int i = 0; i < n; i++) {
        tss += std::pow(yTrue.get(i) - yHat, 2.0);
        rss += std::pow(yTrue.get(i) - yPred.get(i), 2.0);
    }

    // Return R^2
    return 1 - (rss / tss);
}
