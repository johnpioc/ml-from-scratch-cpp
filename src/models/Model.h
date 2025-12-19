#pragma once

#include "../Matrix.h"

struct Model {
public:
    virtual void train(Matrix& trainX, Matrix& trainY) = 0;
    virtual void test(Matrix& testX, Matrix& testY) = 0;

};
