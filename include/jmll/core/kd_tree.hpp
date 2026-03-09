#pragma once

#include <vector>
#include <memory>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>

namespace jmll::core {

class KDTreeNode {
public:
    std::vector<double> data;
    double label;
    std::unique_ptr<KDTreeNode> left = nullptr;
    std::unique_ptr<KDTreeNode> right = nullptr;

    KDTreeNode(std::vector<double>& data, double label);
};

class KDTree {
private:
    std::unique_ptr<KDTreeNode> root_ = nullptr;
    int numOfDimensions_;
    int size_;

public:
    KDTree(Matrix& dataPoints, Vector& labels);

    Vector getKNearest(Vector& dataPoint, int k);
};

}
