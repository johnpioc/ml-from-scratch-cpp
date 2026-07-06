#pragma once

#include <concepts>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/tuning/traits.hpp>

namespace jmll::models::tuning {

template <typename T>
concept FittingPolicy = requires(T policy, core::Matrix& x, core::Vector& y) {
    { policy.fit(x, y) } -> std::same_as<std::pair<core::Vector, int>>;
};

// ===============================================================================================
// LINEAR REGRESSION FITTING POLICIES
// ===============================================================================================

template <typename T>
concept LinearRegressionFittingPolicy = FittingPolicy<T> && forLinearRegression<T> &&
                                        requires(T policy, core::Matrix& x, core::Vector& y) {
                                            {
                                                policy.fit(x, y)
                                            } -> std::same_as<std::pair<core::Vector, int>>;
                                        };

class OLS {
   public:
    std::pair<core::Vector, int> fit(core::Matrix& x, core::Vector& y);
};

template <>
inline constexpr bool forLinearRegression<OLS> = true;

class Ridge {
   private:
    double lambda_;

   public:
    explicit Ridge(double lambda) : lambda_(lambda) {}
    std::pair<core::Vector, int> fit(core::Matrix& x, core::Vector& y);
};

template <>
inline constexpr bool forLinearRegression<Ridge> = true;

}  // namespace jmll::models::tuning
