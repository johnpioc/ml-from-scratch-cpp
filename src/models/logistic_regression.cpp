#include <cmath>
#include <mlfs/core/matrix.hpp>
#include <mlfs/core/vector.hpp>
#include <mlfs/models/logistic_regression.hpp>

using namespace mlfs::core;
using namespace mlfs::models;

void LogisticRegression::fit(Matrix& x, Vector& y) {
    Matrix augmented = x.prependOnes();
    int n = augmented.numRows;

    // Estimate coefficients using gradient descent
    for (int i = 0; i < n; i++) {
    }
}
