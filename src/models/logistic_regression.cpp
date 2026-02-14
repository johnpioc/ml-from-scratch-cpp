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

        Vector derivative = currentRow * error;
        this->beta_ -= derivative * LEARNING_RATE;
    }
}
