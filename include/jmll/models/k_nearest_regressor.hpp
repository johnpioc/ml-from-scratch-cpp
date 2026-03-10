#pragma once

#include "jmll/models/detail/cross_validator.hpp"
#include <jmll/core/lpnorm.hpp>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/core/kd_tree.hpp>
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
        this->space_ = KDTree<DistanceEquation>(x, y);
    }

    Vector predict(Matrix& x) {
        Vector predictions(x.numRows);
        for (int i = 0; i < x.numRows; i++) {
            Vector kNearest = this->space_.getKNearest(x.getRow(i), this->k);
            predictions.set(i, kNearest.mean());
        }
        return predictions;
    }

    double evaluate(Vector& yPred, Vector& yTrue) {
        return this->metric_.evaluate(yPred, yTrue, this->numOfPredictors_);
    }

    double crossValidate(Matrix& x, Vector& y, int numOfFolds) {
        return this->crossValidator_.crossValidate(*this, x, y, numOfFolds);
    }
};

}
