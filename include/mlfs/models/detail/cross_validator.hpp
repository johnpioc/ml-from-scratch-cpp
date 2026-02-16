#pragma once

#include <mlfs/models/tuning/evaluation_metric.hpp>
#include <mlfs/core/matrix.hpp>
#include <mlfs/core/vector.hpp>

namespace mlfs::models::detail {

template <tuning::EvaluationMetric EvaluationMetric>
class CrossValidator {
private:
    EvaluationMetric metric_;

public:
    template <typename... Args>
    double crossValidate(core::Matrix& x, core::Vector& y, int numOfFolds, Args... args);
};
}
