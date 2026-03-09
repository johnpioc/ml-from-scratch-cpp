#pragma once

#include <vector>
#include <memory>

#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/core/lpnorm.hpp>

namespace jmll::core {

class KDTreeNode {
public:
    std::vector<double> data;
    double label;
    std::unique_ptr<KDTreeNode> left = nullptr;
    std::unique_ptr<KDTreeNode> right = nullptr;

    KDTreeNode(std::vector<double>& data, double label);
};

template <DistanceEquation DistanceEquation>
class KDTree {
private:
    std::unique_ptr<KDTreeNode> root_ = nullptr;
    int numOfDimensions_;
    int size_;
    DistanceEquation distanceEquation_;

public:
    KDTree(Matrix& dataPoints, Vector& labels);

    Vector getKNearest(Vector& dataPoint, int k);
};

}
