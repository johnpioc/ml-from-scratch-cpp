#pragma once

#include <concepts>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>

namespace jmll::models::detail {
template <typename T, typename... Args>
concept Model = requires(T model, core::Matrix& x, core::Vector& y, int numOfFolds) {
    { model.fit(x, y) } -> std::same_as<void>;
    { model.predict(x) } -> std::same_as<core::Vector>;
    { model.evaluate(y, y) } -> std::same_as<double>;
    { model.crossValidate(x, y, numOfFolds) } -> std::same_as<double>;
};
}  // namespace jmll::models::detail
