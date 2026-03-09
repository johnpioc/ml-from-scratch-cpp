#pragma once

#include "jmll/models/detail/cross_validator.hpp"
#include <jmll/core/lpnorm.hpp>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/tuning/evaluation_metric.hpp>
#include <jmll/models/tuning/traits.hpp>

using namespace jmll::core;
using namespace jmll::models::tuning;

namespace jmll::models {

template<
    DistanceEquation DistanceEquation = core::Euclidean,
    tuning::RegressionEvaluationMetric EvaluationMetric = tuning::RSquared
>
class KNearestRegressor {
private:
    DistanceEquation distanceEquation_;
    EvaluationMetric metric_;
    KNNStructure<DistanceEquation> space_;
    detail::CrossValidator<EvaluationMetric> crossValidator_;
    int numOfPredictors_;

public:
    int k;

    KNearestRegressor(int k) : k(k) {};

    void fit(Matrix& x, Vector& y) {
        std::pair<KNNStructure<DistanceEquation>, int> results = this->distanceEquation_.fit(x,y);
        this->space_ = results.first;
        this->numOfPredictors_ = results.second;
    }

    Vector predict(Matrix& x) {
        // TODO: implement internal logic in KNN Structure to return vector of predictions
    }

    double evaluate(Vector& yPred, Vector& yTrue) {
        return this->metric_.evaluate(yPred, yTrue, this->numOfPredictors_);
    }

    double crossValidate(Matrix& x, Vector& y, int numOfFolds) {
        return this->crossValidator_.crossValidate(*this, x, y, numOfFolds);
    }
};

}
