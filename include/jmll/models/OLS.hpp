#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>

namespace jmll::models {
using jmll::core::Matrix;
using jmll::core::Vector;

class OLS {
   private:
    Vector coeffs_;

   public:
    /* Default constructor - initialises coefficients to zero */
    OLS();

    /* Fits the model and finds coefficeints with respect to the given input matrix and
     * label vector */
    void fit(Matrix X, Vector y);

    /* Returns a Vector of predictions for a given input matrix */
    Vector predict(Matrix X);
};

}  // namespace jmll::models
