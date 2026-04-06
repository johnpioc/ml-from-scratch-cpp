#pragma once

#include <memory>

#include <jmll/core/lpnorm.hpp>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/core/kd_tree.hpp>

#include <jmll/models/detail/cross_validator.hpp>
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
    detail::CrossValidator<EvaluationMetric> crossValidator_;
    int numOfPredictors_;

    // KD Tree or Ball Tree based on number of dimensions
    std::unique_ptr<KDTree<DistanceEquation>> kdTree_ = nullptr;
    std::unique_ptr<BallTree<DistanceEquation>> ballTree_ = nullptr;

public:
    int k;

    KNearestRegressor(int k) : k(k) {};

    void fit(Matrix& x, Vector& y) {
        this->kdTree_ = std::make_unique<KDTree<DistanceEquation>>(x, y);
    }

    Vector predict(Matrix& x) {
        Vector predictions(x.numRows);
        for (int i = 0; i < x.numRows; i++) {
            Vector currentRow = x.getRow(i);
            Vector kNearest = this->kdTree_->getKNearest(currentRow, this->k);
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
