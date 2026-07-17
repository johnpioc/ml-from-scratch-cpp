#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/ridge.hpp>

namespace jmll::models {
using jmll::core::Matrix;
using jmll::core::Vector;

Ridge::Ridge(double lambda) { this->lambda_ = lambda; }

void Ridge::fit(const Matrix& data, const Vector& labels) {
    Matrix X = data.prependOnes();
    Matrix XTranspose = X.transpose();

    Matrix gram = XTranspose * X;
    Matrix gramPlusLambda = gram + this->lambda_ * core::identity(gram.getNumRows());

    this->coeffs_ = gramPlusLambda.inverse() * XTranspose * labels;
}

Vector Ridge::predict(const Matrix& data) const { return data * this->coeffs_; }

}  // namespace jmll::models
