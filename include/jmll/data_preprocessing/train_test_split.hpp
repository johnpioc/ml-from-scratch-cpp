#pragma once

#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>

namespace jmll::data_preprocessing {

class TrainTestSplit {
   public:
    core::Matrix xTrain{0, 0};
    core::Matrix xTest{0, 0};
    core::Vector yTrain{0};
    core::Vector yTest{0};

    TrainTestSplit(core::Matrix& x, core::Vector& y, double testSplit);
};

}  // namespace jmll::data_preprocessing
