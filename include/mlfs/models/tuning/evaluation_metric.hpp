#pragma once

#include <concepts>
#include <mlfs/core/vector.hpp>
#include <mlfs/models/tuning/traits.hpp>
#include <utility>

namespace mlfs::models::tuning {

template <typename T, typename... Args>
concept EvaluationMetric = requires(T metric, core::Vector& yPred, core::Vector& yTrue, 
    Args&&... args) {
    { metric.evaluate(yPred, yTrue, std::forward<Args>(args)...) } -> std::same_as<double>;
};

// =============================================================================================== 
// LINEAR REGRESSION EVALUATION METRICS
// =============================================================================================== 
//
template <typename T, typename... Args>
concept LinearRegressionEvaluationMetric = 
    EvaluationMetric<T> && forLinearRegression<T> &&
    requires(T metric, core::Vector& yPred, core::Vector& yTrue, Args&&... args) {
        { metric.evaluate(yPred, yTrue, std::forward<Args>(args)...) } -> std::same_as<double>;
};

class RSquared {
public:
    double evaluate(core::Vector& yPred, core::Vector& yTrue);
};

template<>
inline constexpr bool forLinearRegression<RSquared> = true;
}
