#pragma once

#include "../Matrix.h"
#include "Model.h"

struct LinearRegression : Model {
private:
    double coefficient_;
    double intercept_;

public:
    LinearRegression();

    void train(Matrix& trainX, Matrix& trainY);

    void test(Matrix& testX, Matrix& testY);
};
