#pragma once

#include <mlfs/core/vector.hpp>
#include <mlfs/core/matrix.hpp>

namespace mlfs {
namespace models {

class LogisticRegression {
private:
    core::Vector beta_{0};

public:
    void fit(core::Matrix& x, core::Vector& y);

    core::Vector predict(core::Matrix& x);

    double evaluate(core::Vector& yPred, core::Vector& yTrue);
};

}
}
