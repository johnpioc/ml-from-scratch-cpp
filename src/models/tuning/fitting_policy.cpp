#include <mlfs/core/vector.hpp>
#include <mlfs/core/matrix.hpp>
#include <mlfs/models/tuning/fitting_policy.hpp>

using namespace mlfs::core;
using namespace mlfs::models::tuning;

// ===============================================================================================
// LINEAR REGRESSION FITTING POLICIES
// ===============================================================================================
std::pair<Vector, int> OLS::fit(Matrix& x, Vector& y) {
    Matrix augmented = x.prependOnes();
    Matrix XT = augmented.transpose();
    Matrix XTX = XT * augmented;
    Matrix XTX_inv = XTX.inverse();
    return { XTX_inv * XT * y, x.numRows};
}

std::pair<Vector, int> Ridge::fit(Matrix& x, Vector& y) {
    Matrix augmented = x.prependOnes();
    Matrix XT = augmented.transpose();
    Matrix XTX = XT * augmented;

    Matrix lambdaMat = identity(XTX.numRows) * this->lambda_;
    Matrix XTX_lambdaMat = XTX + lambdaMat;
    Matrix XTX_lambdaMat_inv = XTX_lambdaMat.inverse();

    // TODO: create a function that obtains the number of predictors for Ridge Regression
    return { XTX_lambdaMat_inv * XT * y, x.numRows };
}
