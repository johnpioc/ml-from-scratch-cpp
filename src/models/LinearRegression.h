#pragma once

#include "../Matrix.h"

struct LinearRegression {
private:
    size_t numOfPredictors_;
    Matrix coefficients_;
    double intercept_;

public:
    LinearRegression(size_t numOfPredictors);

    void train(Matrix& trainX, Matrix& trainY);

    void test(Matrix& testX, Matrix& testY);
};
