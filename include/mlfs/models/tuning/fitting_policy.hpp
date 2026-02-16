#pragma once

#include <concepts>
#include <mlfs/core/vector.hpp>
#include <mlfs/core/matrix.hpp>
#include <mlfs/models/tuning/traits.hpp>

namespace mlfs::models::tuning {

template<typename T>
concept FittingPolicy = 
    requires(T policy, core::Matrix& x, core::Vector& y) {
        { policy.fit(x, y) } -> std::same_as<core::Vector>;
};

// =============================================================================================== 
// LINEAR REGRESSION FITTING POLICIES
// =============================================================================================== 

template<typename T>
concept LinearRegressionFittingPolicy = 
    FittingPolicy<T> && forLinearRegression<T> &&
    requires(T policy, core::Matrix& x, core::Vector& y) {
    { policy.fit(x, y) } -> std::same_as<core::Vector>;
};

class OLS {
public:
    core::Vector fit(core::Matrix& x, core::Vector& y);
};

template<>
inline constexpr bool forLinearRegression<OLS> = true;

class Ridge {
private:
    const double lambda_;
public:
    explicit Ridge(double lambda) : lambda_(lambda) {}
    core::Vector fit(core::Matrix& x, core::Vector& y);
};

template<>
inline constexpr bool forLinearRegression<Ridge> = true;

}
