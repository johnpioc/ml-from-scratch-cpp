#pragma once

#include <jmll/core/vector.hpp>
#include <concepts>

namespace jmll::core {

template <typename Equation>
concept DistanceEquation = 
    requires (Vector& v) {
        { Equation::calculate(v, v) } -> std::same_as<double>;
};

template <int P>
class LPNorm {
    double calculate(Vector& v1, Vector& v2) {
        double sum = 0.0;
        for (int i = 0; i < v1.numCells; i++) {
            sum += std::pow(std::abs(v1.get(i) - v2.get(i)), P);
        }
        return std::pow(sum, 1.0 / P);
    }
};

using Manhattan = LPNorm<1>;
using Euclidean = LPNorm<2>;
}
