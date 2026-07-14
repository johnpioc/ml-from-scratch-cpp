#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <utility>

namespace jmll::benchmark::data_generation {
using jmll::core::Matrix;
using jmll::core::Vector;

std::pair<Matrix, Vector> makeLinearDataset(int n, int d);
};  // namespace jmll::benchmark::data_generation
