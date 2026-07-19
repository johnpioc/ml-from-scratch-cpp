#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>

namespace jmll::models {
using jmll::core::Matrix;
using jmll::core::Vector;

class KNNRegressor {
   public:
    int k_;
    // Will need to also store KD Tree or Ball Tree here

   private:
    explicit KNNRegressor(int k);

    void fit(const Matrix& data, const Vector& labels);

    [[nodiscard]] Vector predict(const Matrix& data) const;
};
}  // namespace jmll::models
