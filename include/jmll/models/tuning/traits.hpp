#pragma once

#include <jmll/core/kd_tree.hpp>
#include <jmll/core/ball_tree.hpp>
#include <jmll/core/lpnorm.hpp>

namespace jmll::models::tuning {

template <typename T>
inline constexpr bool forRegression = false;

template <typename T>
inline constexpr bool forLinearRegression = false;

template <typename T>
inline constexpr bool forKNearestNeighbours = false;

template<core::DistanceEquation DistanceEquation>
using KNNStructure = std::variant<
    core::KDTree<DistanceEquation>, 
    core::BallTree<DistanceEquation>
>;

}
