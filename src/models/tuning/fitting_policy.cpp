#include <mlfs/core/vector.hpp>
#include <mlfs/core/matrix.hpp>
#include <mlfs/models/tuning/fitting_policy.hpp>

using namespace mlfs::core;
using namespace mlfs::models::tuning;

// ===============================================================================================
// LINEAR REGRESSION FITTING POLICIES
// ===============================================================================================
Vector OLS::fit(Matrix& x, Vector& y) {
    Matrix augmented = x.prependOnes();
    Matrix XT = augmented.transpose();
    Matrix XTX = XT * augmented;
    Matrix XTX_inv = XTX.inverse();
    return XTX_inv * XT * y;
}

Vector Ridge::fit(Matrix& x, Vector& y) {
    return this->fit(x, y, 0.1);
}

Vector Ridge::fit(Matrix& x, Vector& y, double lambda) {
    Matrix augmented = x.prependOnes();
    Matrix XT = augmented.transpose();
    Matrix XTX = XT * augmented;

    Matrix lambdaMat = identity(XTX.numRows);
    Matrix XTX_lambdaMat = XTX + lambdaMat;
    Matrix XTX_lambdaMat_inv = XTX_lambdaMat.inverse();

    return XTX_lambdaMat_inv * XT * y;
}
