#pragma once

#include <concepts>
#include <jmll/core/vector.hpp>
#include <jmll/models/tuning/traits.hpp>

namespace jmll::models::tuning {

template <typename T>
concept EvaluationMetric =
    requires(T metric, core::Vector& yPred, core::Vector& yTrue, int numOfPredictors) {
        { metric.evaluate(yPred, yTrue, numOfPredictors) } -> std::same_as<double>;
    };

// ===============================================================================================
// REGRESSION EVALUATION METRICS
// ===============================================================================================
//
template <typename T>
concept RegressionEvaluationMetric =
    EvaluationMetric<T> && forRegression<T> &&
    requires(T metric, core::Vector& yPred, core::Vector& yTrue, int numOfPredictors) {
        { metric.evaluate(yPred, yTrue, numOfPredictors) } -> std::same_as<double>;
    };

class RSquared {
   public:
    double evaluate(core::Vector& yPred, core::Vector& yTrue, int numOfPredictors);
};

template <>
inline constexpr bool forRegression<RSquared> = true;

class MallowsCp {
   public:
    double evaluate(core::Vector& yPred, core::Vector& yTrue, int numOfPredictors);
};

template <>
inline constexpr bool forRegression<MallowsCp> = true;

class AIC {
   public:
    double evaluate(core::Vector& yPred, core::Vector& yTrue, int numOfPredictors);
};

template <>
inline constexpr bool forRegression<AIC> = true;

class BIC {
   public:
    double evaluate(core::Vector& yPred, core::Vector& yTrue, int numOfPredictors);
};

template <>
inline constexpr bool forRegression<BIC> = true;

class AdjustedRSquared {
   public:
    double evaluate(core::Vector& yPred, core::Vector& yTrue, int numOfPredictors);
};

template <>
inline constexpr bool forRegression<AdjustedRSquared> = true;

}  // namespace jmll::models::tuning
