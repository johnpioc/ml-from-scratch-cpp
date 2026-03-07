#pragma once

#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>

namespace jmll::data_preprocessing {

class TrainTestSplit {
public:
    core::Matrix& xTrain;
    core::Matrix& xTest;
    core::Vector& yTrain;
    core::Vector& yTest;

    TrainTestSplit(core::Matrix& x, core::Vector& y, double testSplit);
};

}
