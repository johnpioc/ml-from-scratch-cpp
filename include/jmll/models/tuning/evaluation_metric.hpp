#pragma once

#include <concepts>
#include <jmll/core/vector.hpp>
#include <jmll/models/tuning/traits.hpp>
#include <utility>

namespace jmll::models::tuning {

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
