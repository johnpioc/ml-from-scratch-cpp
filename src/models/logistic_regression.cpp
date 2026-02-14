#include <cmath>
#include <mlfs/core/matrix.hpp>
#include <mlfs/core/vector.hpp>
#include <mlfs/models/logistic_regression.hpp>

#define LEARNING_RATE 0.00001

using namespace mlfs::core;
using namespace mlfs::models;

void LogisticRegression::fit(Matrix& x, Vector& y) {
    Matrix augmented = x.prependOnes();
    int n = augmented.numRows;

    // Estimate coefficients using gradient descent
    for (int i = 0; i < n; i++) {
        Vector currentRow = augmented.getRow(i);
        double exponential = std::exp(this->beta_ * currentRow);
        double estimate = exponential / (1 + exponential);
        double error = y.get(i) - estimate;

        Vector derivative = currentRow * error * LEARNING_RATE;
        this->beta_ -= derivative;
    }
}

Vector LogisticRegression::predict(Matrix& x) {
    Matrix augmented = x.prependOnes();
    Vector logit = augmented * this->beta_;

    Vector yPred(logit.numCells);

    // Apply exponential function to logit vector values and get prediction
    for (int i = 0; i < logit.numCells; i++) {
        double exponential = std::exp(logit.get(i));
        double estimate = exponential / (1 + exponential);
        yPred.set(i, estimate > 0.5 ? 1 : 0);
    }

    return yPred;
}

double LogisticRegression::evaluate(Vector& yPred, Vector& yTrue) {
    int n = yPred.numCells;
    int numErrors = 0;

    for (int i = 0; i < n; i++) {
        if (yPred.get(i) != yTrue.get(i)) numErrors++;
    }

    return (double) numErrors / (double) n;
}
