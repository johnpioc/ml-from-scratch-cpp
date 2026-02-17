#pragma once

#include <concepts>
#include <mlfs/core/vector.hpp>
#include <mlfs/models/tuning/traits.hpp>
#include <utility>

namespace mlfs::models::tuning {

template <typename T>
concept EvaluationMetric = requires(T metric, core::Vector& yPred, core::Vector& yTrue,
    int numOfPredictors) {
    { metric.evaluate(yPred, yTrue, numOfPredictors) } -> std::same_as<double>;
};

// =============================================================================================== 
// LINEAR REGRESSION EVALUATION METRICS
// =============================================================================================== 
//
template <typename T>
concept LinearRegressionEvaluationMetric = 
    EvaluationMetric<T> && forLinearRegression<T> &&
    requires(T metric, core::Vector& yPred, core::Vector& yTrue, int numOfPredictors) {
        { metric.evaluate(yPred, yTrue, numOfPredictors) } -> std::same_as<double>;
};

class RSquared {
public:
    double evaluate(core::Vector& yPred, core::Vector& yTrue, int numOfPredictors);
};

template<>
inline constexpr bool forLinearRegression<RSquared> = true;
}
