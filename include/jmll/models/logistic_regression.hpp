#pragma once

#include <jmll/core/vector.hpp>
#include <jmll/core/matrix.hpp>

namespace jmll {
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
