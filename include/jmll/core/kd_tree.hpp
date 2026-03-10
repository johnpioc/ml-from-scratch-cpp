#pragma once

#include <vector>
#include <memory>
#include <queue>

#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/core/lpnorm.hpp>

namespace jmll::core {

class KDTreeNode {
public:
    Vector point;
    double label;
    int axis;
    std::unique_ptr<KDTreeNode> left = nullptr;
    std::unique_ptr<KDTreeNode> right = nullptr;

    KDTreeNode(Vector& point, double label);
};

template <DistanceEquation DistanceEquation>
class KDTree {
private:
    std::unique_ptr<KDTreeNode> root_ = nullptr;
    int numOfDimensions_;
    int size_;
    DistanceEquation distanceEquation_;

    std::unique_ptr<KDTreeNode> build(Matrix& points, Vector& labels, std::vector<int>& indices,
        int depth);

    void search(KDTreeNode* node, Vector& target, int k, 
        std::priority_queue<std::pair<double,double>>& topK);

public:
    KDTree(Matrix& dataPoints, Vector& labels);

    Vector getKNearest(Vector& dataPoint, int k);
};

}
