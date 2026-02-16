#include <mlfs/models/detail/cross_validator.hpp>
#include <mlfs/models/tuning/evaluation_metric.hpp>
#include <mlfs/core/matrix.hpp>
#include <mlfs/core/vector.hpp>

using namespace mlfs::models::detail;
using namespace mlfs::models::tuning;
using namespace mlfs::core;

template <EvaluationMetric EvaluationMetric>
template <typename... Args>
double CrossValidator<EvaluationMetric>::crossValidate(Matrix& x, Vector& y, int numOfFolds,
    Args... args) {
    // Seperate data into k folds
    // For each k fold, call fit and evaluate()
    // Return the average of all evaluation metrics for each k fold
}
