#pragma once

#include "../Matrix.h"
#include "Model.h"

struct LogisticRegression : Model {
private:
    double intercept_;
    double coefficient_;

public:
    LogisticRegression();

    void train(Matrix& trainX, Matrix& trainY);

    void test(Matrix& testX, Matrix& testY);
};
