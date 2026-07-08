#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/OLS.hpp>

namespace jmll::models {
using jmll::core::Matrix;
using jmll::core::Vector;

OLS::OLS() { this->coeffs_ = Vector(0); }
}  // namespace jmll::models
