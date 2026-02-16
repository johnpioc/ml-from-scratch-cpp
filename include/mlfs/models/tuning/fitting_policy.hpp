#pragma once

#include <mlfs/core/vector.hpp>
#include <mlfs/core/matrix.hpp>
#include <mlfs/models/tuning/traits.hpp>
#include <utility>

namespace mlfs::models::tuning {

template<typename T, typename... Args>
concept FittingPolicy = requires(T policy, core::Matrix& x, core::Vector& y, 
    Args&&... args) {
    { policy.fit(x, y, std::forward<Args>(args)...) } -> std::same_as<core::Vector>;
};

// =============================================================================================== 
// LINEAR REGRESSION FITTING POLICIES
// =============================================================================================== 

template<typename T, typename... Args>
concept LinearRegressionFittingPolicy = 
    FittingPolicy<T> && forLinearRegression<T> &&
    requires(T policy, core::Matrix& x, core::Vector& y, Args&&... args) {
    { policy.fit(x, y, std::forward<Args>(args)...) } -> std::same_as<core::Vector>;
};

class OLS {
public:
    core::Vector fit(core::Matrix& x, core::Vector& y);
};

template<>
inline constexpr bool forLinearRegression<OLS> = true;

class Ridge {
public:
    core::Vector fit(core::Matrix& x, core::Vector& y);
    core::Vector fit(core::Matrix& x, core::Vector& y, double lambda);
};

template<>
inline constexpr bool forLinearRegression<Ridge> = true;

}
