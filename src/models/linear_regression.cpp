#include <cmath>
#include <mlfs/core/matrix.hpp>
#include <mlfs/core/vector.hpp>
#include <mlfs/models/linear_regression.hpp>

using namespace mlfs::models;
using namespace mlfs::core;

void LinearRegression::fit(Matrix& x, Vector& y) {
    // Augment x to include a column of 1.0s at the start to represent intercept
    Matrix augmented(x.numRows, x.numCols + 1);
    for (int r = 0; r < augmented.numRows; r++) {
         for (int c = 0; c < augmented.numCols; c++) {
            augmented.set(r, c, c == 0 ? 1.0 : x.get(r, c - 1));
        }
    }

    // Get OLS solution
    Matrix XT = augmented.transpose();
    Matrix XTX = XT * augmented;
    Matrix XTX_inv = XTX.inverse();
    this->beta_ = XTX_inv * XT * y;
}

Vector LinearRegression::predict(Matrix& x) {
    // Augment x to include a column of 1.0s at the start to represent intercept
    Matrix augmented(x.numRows, x.numCols + 1);
    for (int r = 0; r < augmented.numRows; r++) {
         for (int c = 0; c < augmented.numCols; c++) {
            augmented.set(r, c, c == 0 ? 1.0 : x.get(r, c - 1));
        }
    }

    return augmented * this->beta_;
}

double LinearRegression::evaluate(Vector& yPred, Vector& yTrue) {
    int n = yPred.numCells;

    // Calculate mean response
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += yTrue.get(i);
    }
    double yHat = sum / n;

    // Calculate tss and rss
    double tss = 0, rss = 0;
    for (int i = 0; i < n; i++) {
        tss += std::pow(yTrue.get(i) - yHat, 2.0);
        rss += std::pow(yTrue.get(i) - yPred.get(i), 2.0);
    }

    // Return R^2
    return 1 - (rss / tss);
}
