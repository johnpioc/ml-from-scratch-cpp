#include <algorithm>
#include <jmll/core/kd_tree.hpp>
#include <jmll/core/vector.hpp>
#include <memory>
#include <vector>

namespace jmll::core {
// ===============================================================================================
// KD NODE METHODS
// ===============================================================================================
KDNode::KDNode(const Vector& key, double value) {
    this->key_ = key;
    this->value_ = value;
}

void KDNode::setLeft(std::shared_ptr<KDNode> node) { this->left_ = node; }
std::shared_ptr<KDNode> KDNode::getLeft() { return this->left_; }

void KDNode::setRight(std::shared_ptr<KDNode> node) { this->right_ = node; }
std::shared_ptr<KDNode> KDNode::getRight() { return this->right_; }

int KDNode::getDimIndex() const { return this->dimIndex_; }
void KDNode::setDimIndex(int dimIndex) { this->dimIndex_ = dimIndex; }

int KDNode::getKeySize() const { return this->key_.getNumCells(); }
double KDNode::getKeyVal(int i) const { return this->key_.get(i); }

// ===============================================================================================
// KD TREE METHODS
// ===============================================================================================
std::shared_ptr<KDNode> build(std::vector<std::shared_ptr<KDNode>> nodes, int dimIndex) {
    // Get median of node array for current dimension
    std::sort(nodes.begin(), nodes.end(),
              [dimIndex](std::shared_ptr<KDNode> a, std::shared_ptr<KDNode> b) {
                  return a->getKeyVal(dimIndex) < b->getKeyVal(dimIndex);
              });

    int medianIndex = nodes.size() / 2;

    /* Partition node array (without the median node) into:
     * left: an array w/ dimension value less than the median node
     * right: an array w/ dimension value greater than or equal to the median node
     */
    std::vector<std::shared_ptr<KDNode>> left;
    std::vector<std::shared_ptr<KDNode>> right;
    for (int i = 0; i < nodes.size(); i++) {
        if (i < medianIndex) {
            left.push_back(nodes[i]);
        } else if (i > medianIndex) {
            right.push_back(nodes[i]);
        }
    }

    int nextDimIndex = (dimIndex + 1) % nodes[medianIndex]->getKeySize();

    if (left.size() > 0) {
        nodes[medianIndex]->setLeft(build(left, nextDimIndex));
    }

    if (right.size() > 0) {
        nodes[medianIndex]->setRight(build(right, nextDimIndex));
    }

    // Return median
    return nodes[medianIndex];
}

KDTree::KDTree(const Matrix& data, const Vector& labels) {
    this->size_ = data.getNumRows();

    // Build a vector of KD nodes
    std::vector<std::shared_ptr<KDNode>> nodes;

    for (int r = 0; r < this->size_; r++) {
        KDNode node(data.getRow(r), labels.get(r));
        nodes.push_back(std::make_shared<KDNode>(node));
    }

    // Build the KD tree
    this->root_ = build(nodes, 0);
}
}  // namespace jmll::core
