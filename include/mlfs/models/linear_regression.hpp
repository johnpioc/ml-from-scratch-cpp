#pragma once

#include "mlfs/models/tuning/evaluation_metric.hpp"
#include <mlfs/core/matrix.hpp>
#include <mlfs/core/vector.hpp>
#include <mlfs/models/tuning/fitting_policy.hpp>

namespace mlfs {
namespace models {

template<
    tuning::LinearRegressionFittingPolicy FittingPolicy = tuning::OLS,
    tuning::LinearRegressionEvaluationMetric EvaluationMetric = tuning::RSquared
>
class LinearRegression {
private:
    core::Vector beta_{0};
    FittingPolicy policy_;
    EvaluationMetric metric_;

public:
    /* Takes a given matrix of observation predictors and a given vector of observation responses
     * and performs ordinary least squares to calculate model intercept and coefficients
     * */
    template<typename... Args>
    void fit(core::Matrix& x, core::Vector& y, Args... args) { 
        this->beta_ = this->policy_.fit(x, y, args...);
    }

    /* Takes a given matrix of observation and predicts a vector of responses based on 
     * the trained intercept and coefficient values
     * */
    core::Vector predict(core::Matrix& x) {
        core::Matrix augmented = x.prependOnes();
        return augmented * this->beta_;
    }

    template<typename... Args>
    double evaluate(core::Vector& yPred, core::Vector& yTrue, Args... args) {
        return this->metric_.evaluate(yPred, yTrue, args...);
    }
};

}
}
