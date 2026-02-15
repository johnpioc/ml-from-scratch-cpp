#pragma once

#include <mlfs/core/vector.hpp>
#include <mlfs/core/matrix.hpp>
#include <utility>

namespace mlfs::models::tuning {

template<typename T, typename... Args>
concept LinearRegressionFittingPolicy = requires(T policy, core::Matrix& x, core::Vector& y, 
    Args&&... args) {
    { policy.fit(x, y, std::forward<Args>(args)...) } -> std::same_as<core::Vector>;
};

class OLS {
public:
    core::Vector fit(core::Matrix& x, core::Vector& y);
};

class Ridge {
public:
    core::Vector fit(core::Matrix& x, core::Vector& y);
    core::Vector fit(core::Matrix& x, core::Vector& y, double lambda);
};

}
