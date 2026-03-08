#pragma once

#include <concepts>
#include <jmll/core/vector.hpp>
#include <jmll/core/matrix.hpp>
#include <jmll/core/ball_tree.hpp>
#include <jmll/core/kd_tree.hpp>
#include <jmll/models/tuning/traits.hpp>

namespace jmll::models::tuning {

template<typename Policy, typename ParamDataStructure>
concept FittingPolicy = 
    requires(Policy policy, core::Matrix& x, core::Vector& y) {
        { policy.fit(x, y) } -> std::same_as<std::pair<ParamDataStructure, int>>;
};

// =============================================================================================== 
// LINEAR REGRESSION FITTING POLICIES
// =============================================================================================== 

template<typename T>
concept LinearRegressionFittingPolicy = 
    FittingPolicy<T, core::Vector> && forLinearRegression<T> &&
    requires(T policy, core::Matrix& x, core::Vector& y) {
    { policy.fit(x, y) } -> std::same_as<std::pair<core::Vector, int>>;
};

class OLS {
public:
    std::pair<core::Vector, int> fit(core::Matrix& x, core::Vector& y);
};

template<>
inline constexpr bool forLinearRegression<OLS> = true;

class Ridge {
private:
    double lambda_;
public:
    explicit Ridge(double lambda) : lambda_(lambda) {}
    std::pair<core::Vector, int> fit(core::Matrix& x, core::Vector& y);
};

template<>
inline constexpr bool forLinearRegression<Ridge> = true;

// =============================================================================================== 
// K NEAREST NEIGHBOURS DISTANCE EQUATIONS
// =============================================================================================== 
using KNNStructure = std::variant<core::KDTree, core::BallTree>;

template<typename DistanceEquation>
concept KNNDistanceEquation = 
    FittingPolicy<DistanceEquation, KNNStructure> && forKNearestNeighbours<DistanceEquation> &&
    requires (DistanceEquation policy, core::Matrix& x, core::Vector& y) {
        { policy.fit(x, y) } -> std::same_as<std::pair<KNNStructure, int>>;
};

class Manhattan {
public:
    std::pair<KNNStructure, int> fit(core::Matrix& x, core::Vector& y);
};

template<>
inline constexpr bool forKNearestNeighbours<Manhattan> = true;

class Euclidean {
public:
    std::pair<KNNStructure, int> fit(core::Matrix& x, core::Vector& y);
};

template<>
inline constexpr bool forKNearestNeighbours<Euclidean> = true;
}
