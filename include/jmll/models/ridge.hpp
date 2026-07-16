#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>

namespace jmll::models {
using jmll::core::Matrix;
using jmll::core::Vector;

class Ridge {
   private:
    Vector coeffs_;
    double lambda_;

   public:
    explicit Ridge(double lambda);

    void fit(const Matrix& data, const Vector& labels);

    [[nodiscard]] Vector predict(const Matrix& data) const;
};
}  // namespace jmll::models
