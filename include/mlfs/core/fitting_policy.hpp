#pragma once

#include <mlfs/core/vector.hpp>
#include <mlfs/core/matrix.hpp>

#include <utility>
#include <vector>
namespace mlfs::core {

template<typename T, typename... Args>
concept LinearRegressionFittingPolicy = requires(T model, Matrix& x, Vector& y, Args&&... args) {
    { model.fit(x, y, std::forward<Args>(args)...) } -> std::same_as<Vector>;
};

class OLS {
public:
    Vector fit(Matrix& x, Vector& y);
};

class Ridge {
private:
    double lambda_;
public:
    Vector fit(Matrix& x, Vector& y, double lambda);
    Vector fit(Matrix& x, Vector& y, std::vector<double> lambda);
};

}
