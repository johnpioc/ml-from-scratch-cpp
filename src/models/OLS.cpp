#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/OLS.hpp>

namespace jmll::models {
using jmll::core::Matrix;
using jmll::core::Vector;

OLS::OLS() { this->coeffs_ = Vector(0); }

void OLS::fit(const Matrix& X, const Vector& y) {
    // Augment input matrix with a column on 1.0's for the bias term
    Matrix XAugmented = X.prependOnes();

    // Solve coefficients imperically using linear algebra
    Matrix XTranpose = XAugmented.transpose();
    Matrix XTranposeXInverse = (XTranpose * XAugmented).inverse();

    this->coeffs_ = XTranposeXInverse * XTranpose * y;
}

Vector OLS::predict(const Matrix& X) {
    // Return zero vector if coefficients haven't been solved for
    if (this->coeffs_.getNumCells() == 0) {
        return Vector(X.getNumRows());
    }

    return X * this->coeffs_;
}
}  // namespace jmll::models
