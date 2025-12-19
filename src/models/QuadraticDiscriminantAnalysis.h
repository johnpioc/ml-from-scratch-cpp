#pragma once

#include <vector>

#include "Model.h"
#include "../Matrix.h"

struct QuadraticDiscriminantAnalysis : Model {
private:
    int numOfPredictors_;
    Matrix prior_;
    std::vector<Matrix> means_;
    std::vector<Matrix> covariances_;

public:
    QuadraticDiscriminantAnalysis(int numOfPredictors);

    void train(Matrix& trainX, Matrix& trainY);

    void test(Matrix& testX, Matrix& testY);
};
