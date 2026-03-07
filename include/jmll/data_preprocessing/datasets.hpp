#pragma once

#include <jmll/data_preprocessing/train_test_split.hpp>

namespace jmll::data_preprocessing {
    TrainTestSplit getBostonData(double testSplit);
    TrainTestSplit getStockMarketData(double testSplit);
}
