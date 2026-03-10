#include "jmll/core/lpnorm.hpp"
#include <jmll/core/kd_tree.hpp>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>

#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>
#include <queue>
#include <utility>

using namespace jmll::core;

// ==============================================================================================
// KD TREE / NODE
// ==============================================================================================
KDTreeNode::KDTreeNode(Vector& point, double label): point(point), label(label) {};

template <DistanceEquation DistanceEquation>
std::unique_ptr<KDTreeNode> KDTree<DistanceEquation>::build(Matrix& points, Vector& labels, 
    std::vector<int>& indices, int depth) {
    if (indices.empty()) return nullptr;

    // Get median
    int axis = depth % points.numCols;
    std::sort(indices.begin(), indices.end(), [&](const int a, const int b) {
        return points.get(a, axis) < points.get(b, axis);
    });
    int medianIndex = indices[indices.size() / 2];

    // Create current node
    std::unique_ptr<KDTreeNode> node = std::make_unique<KDTreeNode>(
        points.getRow(medianIndex).getData(), 
        labels.get(medianIndex)
    );
    node->axis = axis;

    // Partition the indices into two halves based on median
    std::vector<int> leftIndices;
    std::vector<int> rightIndices;

    double median = points.get(medianIndex, axis);
    for (int index : indices) {
        if (index == medianIndex) continue;

        if (points.get(index, axis) < median) leftIndices.push_back(index);
        else rightIndices.push_back(index);
    }

    // Build the left and right child
    node->left = build(points, labels, leftIndices, depth + 1);
    node->right = build(points, labels, rightIndices, depth + 1);

    return node;
}

template <DistanceEquation DistanceEquation>
void KDTree<DistanceEquation>::search(KDTreeNode* node, Vector& target, int k, 
    std::priority_queue<std::pair<double, double>>& topK) {
    if (node == nullptr) return;

    // Calculate distance to current node
    double distance = this->distanceEquation_.calculate(node->point, target);

    // Update max heap
    if (topK.size() < k) {
        topK.push({ distance, node->label });
    } else if (topK.top().first > distance) {
        topK.pop();
        topK.push({ distance, node->label });
    }

    // Determine whether to visit right or left first
    int axis = node->axis;
    double splitValue = node->point.get(axis);

    KDTreeNode* nearChild = target.get(axis) < splitValue 
        ? node->left.get() : node->right.get();
    KDTreeNode* farChild = target.get(axis) < splitValue 
        ? node->right.get() : node->left.get();

    // Visit near side first
    search(nearChild, target, k, topK);

    // Check if we need to visit far child
    double distanceToFar = std::abs(target.get(axis) - node->point.get(axis));

    if (topK.size() < k || distanceToFar < topK.top().first)
        search(farChild, target, k, topK);

}

template <DistanceEquation DistanceEquation>
KDTree<DistanceEquation>::KDTree(Matrix& dataPoints, Vector& labels) {
    this->numOfDimensions_ = dataPoints.numCols;
    this->size_ = dataPoints.numRows;

    std::vector<int> indices(dataPoints.numRows);
    std::iota(indices.begin(), indices.end(), 0);

    this->root_ = build(dataPoints, labels, indices, 0);
}

template <DistanceEquation DistanceEquation>
Vector KDTree<DistanceEquation>::getKNearest(Vector& dataPoint, int k) {
    // First is distance, second is label
    std::priority_queue<std::pair<double,double>> topK;

    // Search for k nearest neighbours
    this->search(this->root_.get(), dataPoint, k, topK);

    // Initialise a Vector to store k nearest neighbours
    Vector kNearest(k);

    int i = 0;
    while (!topK.empty()) {
        kNearest.set(i, topK.top().second);
        topK.pop();
    }

    return kNearest;
}


