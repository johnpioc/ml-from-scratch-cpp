#pragma once

namespace jmll::models::tuning {

template <typename T>
inline constexpr bool forRegression = false;

template <typename T>
inline constexpr bool forLinearRegression = false;

template <typename T>
inline constexpr bool forKNearestNeighbours = false;
}
