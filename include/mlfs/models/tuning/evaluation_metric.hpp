#pragma once

#include <concepts>
#include <mlfs/core/vector.hpp>
#include <utility>

namespace mlfs::models::tuning {

template <typename T, typename... Args>
concept LinearRegressionEvaluationMetric = requires(T metric, core::Vector& yPred, 
    core::Vector& yTrue, Args&&... args) {
        { metric.evaluate(yPred, yTrue, std::forward<Args>(args)...) } -> std::same_as<double>;
};

class RSquared {
public:
    double evaluate(core::Vector& yPred, core::Vector& yTrue);
};

}
