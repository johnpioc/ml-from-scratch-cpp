#include <jmll/data_preprocessing/train_test_split.hpp>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>

#include <vector>
#include <numeric>

using namespace jmll::data_preprocessing;
using namespace jmll::core;

TrainTestSplit::TrainTestSplit(Matrix& x, Vector& y, double testSplit) {
    int n = x.numRows;
    int testSplitIndex = n - n * testSplit;

    int numOfTrainRows = testSplitIndex;
    int numOfTestRows = n - testSplitIndex;

    std::vector<int> trainIndices(numOfTrainRows);
    std::vector<int> testIndices(numOfTestRows);

    std::iota(trainIndices.begin(), trainIndices.end(), 0);
    std::iota(testIndices.begin(), testIndices.end(), testSplitIndex);

    this->xTrain = x.getRows(trainIndices);
    this->xTest = x.getRows(testIndices);
    this->yTrain = Vector(y.getDataByIndices(trainIndices));
    this->yTest = Vector(y.getDataByIndices(testIndices));
}
