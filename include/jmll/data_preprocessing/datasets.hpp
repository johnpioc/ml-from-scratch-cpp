#pragma once

#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>

namespace jmll::data_preprocessing {

class TrainTestSplit {
public:
    const core::Matrix& xTrain;
    const core::Matrix& xTest;
    const core::Vector& yTrain;
    const core::Vector& yTest;

    TrainTestSplit(core::Matrix& x, core::Vector& y, double testSplit);
};

}
