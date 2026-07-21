#include <algorithm>
#include <jmll/core/kd_tree.hpp>
#include <jmll/core/vector.hpp>
#include <memory>
#include <queue>
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

const Vector KDNode::getKey() const { return this->key_; }

// ===============================================================================================
// KD TREE METHODS
// ===============================================================================================
std::shared_ptr<KDNode> build(std::vector<std::shared_ptr<KDNode>> nodes, int dimIndex) {
    // Get median of node array for current dimension
    std::sort(nodes.begin(), nodes.end(),
              [dimIndex](const std::shared_ptr<KDNode>& a, const std::shared_ptr<KDNode>& b) {
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

// Custom priority queue used for searching nearest neighbours
class TopKPriorityQueue {
   private:
    Vector targetPoint_;
    int k_;
    double closestDist_ = INT_MAX;

    struct DistanceComparator {
        Vector targetPoint;

        explicit DistanceComparator(const Vector& targetPoint) : targetPoint(targetPoint) {}

        bool operator()(const std::shared_ptr<KDNode>& a, const std::shared_ptr<KDNode>& b) const {
            // Use euclidean distance
            return euclideanDistance(a, targetPoint) < euclideanDistance(b, targetPoint);
        }
    };

    std::priority_queue<std::shared_ptr<KDNode>, std::vector<std::shared_ptr<KDNode>>,
                        DistanceComparator>
        pq_;

   public:
    TopKPriorityQueue(const Vector& targetPoint, int k) : targetPoint_(targetPoint), k_(k) {
        this->pq_ =
            std::priority_queue<std::shared_ptr<KDNode>, std::vector<std::shared_ptr<KDNode>>,
                                DistanceComparator>(DistanceComparator(targetPoint));
    }

    void insert(std::shared_ptr<KDNode>& node) {
        // Calculate distance to target point
        double dist = euclideanDistance(node, this->targetPoint_);

        // If priority queue isn't full, insert instantly
        if (this->pq_.size() < this->k_) {
            this->pq_.push(node);
        } else {
            // If priority queue IS full, check that distance is smaller than farthest distance in
            // pq
            double worstDist = euclideanDistance(this->pq_.top(), this->targetPoint_);

            if (worstDist > dist) {
                this->pq_.pop();
                this->pq_.push(node);
            }
        }

        // Recheck closest distance
        this->closestDist_ = std::max(dist, this->closestDist_);
    }

    // warning: clears whole data structure
    [[nodiscard]] std::vector<std::shared_ptr<KDNode>> topK() {
        std::vector<std::shared_ptr<KDNode>> res;
        for (int i = 0; i < k_; i++) {
            res.push_back(this->pq_.top());
            this->pq_.pop();
        }

        return res;
    }

    [[nodiscard]] double getClosestDistance() const { return this->closestDist_; }

    // Distance Equations
    [[nodiscard]] static double euclideanDistance(const std::shared_ptr<KDNode>& a,
                                                  const Vector& b) {
        Vector delta = a->getKey() - b;
        for (int i = 0; i < delta.getNumCells(); i++) {
            delta.set(i, delta.get(i) * delta.get(i));
        }
        return delta.getSum();
    }

    [[nodiscard]] static double euclideanDistance(const std::shared_ptr<KDNode>& a,
                                                  const std::shared_ptr<KDNode>& b) {
        Vector delta = a->getKey() - b->getKey();
        for (int i = 0; i < delta.getNumCells(); i++) {
            delta.set(i, delta.get(i) * delta.get(i));
        }
        return delta.getSum();
    }
};

void traverse(std::shared_ptr<KDNode> current, int dimIndex, const Vector& target,
              TopKPriorityQueue& pq) {
    // Exit if current doesn't exist
    if (current == nullptr) return;

    // Go left or right depending on whether the point is lesser than or greater than the current
    // node in the split dimension
    int currentSplitVal = current->getKeyVal(dimIndex);
    int targetSplitVal = target.get(dimIndex);
    int newDimIndex = (dimIndex + 1) % target.getNumCells();
    double currentDist = TopKPriorityQueue::euclideanDistance(current, target);

    if (currentSplitVal < targetSplitVal) {  // Go left
        traverse(current->getLeft(), newDimIndex, target, pq);

        // Traverse right side if current dist is closer than best dist found so far
        if (currentDist <= pq.getClosestDistance()) {
            traverse(current->getRight(), newDimIndex, target, pq);
        }
    } else {
        traverse(current->getRight(), newDimIndex, target, pq);

        // Traverse left side if current dist is closer than best dist found so far
        if (currentDist <= pq.getClosestDistance()) {
            traverse(current->getLeft(), newDimIndex, target, pq);
        }
    }

    // Insert current into priority queue (will be rejected if distance to target is less than worst
    // distance seen so far)
    pq.insert(current);
}

Vector KDTree::searchNearestNeighbours(const Vector& point, int k) const {
    TopKPriorityQueue pq(point, k);
    traverse(this->root_, 0, point, pq);
    std::vector<std::shared_ptr<KDNode>> topK = pq.topK();

    Vector res(k);
    for (int i = 0; i < k; i++) {
        res.set(i, topK[i]->getValue());
    }
    return res;
}

}  // namespace jmll::core
