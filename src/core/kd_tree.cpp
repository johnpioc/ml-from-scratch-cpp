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
// HELPERS
// ==============================================================================================
std::unique_ptr<KDTreeNode> 
build(Matrix& points, Vector& labels, std::vector<int>& indices, int depth) {
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

void search(std::unique_ptr<KDTreeNode>& node, Vector& target, int k, 
    std::priority_queue<std::pair<double, double>>& topK) {
    if (node == nullptr) return;

    // Calculate distance to current node
}

// ==============================================================================================
// KD TREE / NODE FUNCTIONS
// ==============================================================================================
KDTreeNode::KDTreeNode(std::vector<double>& data, double label): data(data), label(label) {};

KDTree::KDTree(Matrix& dataPoints, Vector& labels) {
    this->numOfDimensions_ = dataPoints.numCols;
    this->size_ = dataPoints.numRows;

    std::vector<int> indices(dataPoints.numRows);
    std::iota(indices.begin(), indices.end(), 0);

    this->root_ = build(dataPoints, labels, indices, 0);
}

Vector KDTree::getKNearest(Vector& dataPoint, int k) {

}


