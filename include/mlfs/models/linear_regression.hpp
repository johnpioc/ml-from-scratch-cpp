#pragma once

#include <mlfs/core/matrix.hpp>
#include <mlfs/core/vector.hpp>
#include <mlfs/core/fitting_policy.hpp>

namespace mlfs {
namespace models {

template<core::LinearRegressionFittingPolicy FittingPolicy>
class LinearRegression {
private:
    core::Vector beta_{0};
    FittingPolicy policy_;

public:
    /* Takes a given matrix of observation predictors and a given vector of observation responses
     * and performs ordinary least squares to calculate model intercept and coefficients
     * */
    template<typename... Args>
    void fit(core::Matrix& x, core::Vector& y, Args... args);

    /* Takes a given matrix of observation and predicts a vector of responses based on 
     * the trained intercept and coefficient values
     * */
    core::Vector predict(core::Matrix& x);

    /* Takes a given vector of predicted responses and true responses and returns a R^2 value */
    double evaluate(core::Vector& yPred, core::Vector& yTrue);
};

}
}
