#pragma once

#include <mlfs/core/matrix.hpp>
#include <vector>

namespace mlfs {
namespace models {

class LinearRegression {
private:
    double intercept_;
    std::vector<double> coefficients_;

public:
    void fit(core::Matrix& x, std::vector<double>& y);
    std::vector<double> predict(core::Matrix& x);
    double evaluate(std::vector<double>& yPred, std::vector<double>& yTrue);
};

}
}
