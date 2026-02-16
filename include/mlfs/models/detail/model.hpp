#pragma once

#include <concepts>
#include <mlfs/core/matrix.hpp>
#include <mlfs/core/vector.hpp>

namespace mlfs::models::detail {
template <typename T, typename... Args>
concept Model = requires(T model, core::Matrix& x, core::Vector& y, Args&&... args) {
    { model.fit(x, y) } -> std::same_as<void>;
    { model.predict(x) } -> std::same_as<core::Vector>;
    { model.evaluate(y, y, args...) } -> std::same_as<double>;
};
}
