#pragma once

#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/core/lpnorm.hpp>

#include <vector>
#include <memory>

namespace jmll::core {

class BallTreeNode {
public:
    Vector centroid;
    double radius = 0.0;
    std::vector<int> dataIndices;
    std::unique_ptr<BallTreeNode> left = nullptr;
    std::unique_ptr<BallTreeNode> right = nullptr;

    bool isLeafNode() {
        return left == nullptr && right == nullptr;
    }
};

template <DistanceEquation DistanceEquation>
class BallTree {
private:
    int numOfDimensions_;
    int size_
    DistanceEquation distanceEquation_;
    const int LEAF_SIZE = 30;

    Matrix dataPoints_;
    Vector labels_;

    std::unique_ptr<BallTreeNode> build(vector<int>& indices) {
        std::unique_ptr<BallTreeNode> currNode = make_unique<BallTreeNode>();
        Matrix& currentData = this->dataPoints_.getRows(indices);

        // Calculate Centroid using means
        currNode->centroid = currentData.getRowMeans();

        // Get radius using distance between centroid and furthest point in space
        for (int i : indices) {
            int currDistance = this->distanceEquation_.calculate(
                currNode->centroid, this->dataPoints_.getRow(i)
            );
            currNode->radius = max(currNode->radius, currDistance);
        }
        
        // If number of remaining points is low, make a leaf node and return it
        if (indices.size() <= LEAF_SIZE) {
            currNode->dataIndices = indices;
        } else {
            // Find Dimension with largest spread via variance
            Vector colVariances = this->dataPoints_.getRows(indices).getColVariances();
            std::vector colVariancesData = colVariances.getData();
            int splitDimIdx = std::distance(
                colVariancesData.begin(),
                std::max_element(colVariancesData.begin(), colVariancesData.end())
            );

            // Partition remaining data points based on if they are lower or higher than median
            std::vector<int> leftIndices;
            std::vector<int> rightIndices;

            double dimMedian = this->dataPoints_.getRows(indices).getCol(splitDimIdx).median();
            for (int i : indices) {
                if (this->dataPoints_.get(i, splitDimIdx) <= dimMedian)
                    leftIndices.push_back(i);
                else rightIndices.push_back(i);
            }

            currNode->left = this->build(leftIndices);
            currNode->right = this->build(rightIndices);
        }

        return currNode;
    }

    void search(BallTreeNode* node, Vector& target, int k, 
        std::priority_queue<std::pair<double, double>>& topK) {
        if (node == nullptr) return;

        if (node->isLeafNode()) {
            for (int idx : node->dataIndices) {
                double currentDist = 
                    this->distanceEquation_.calculate(target, this->dataPoints_.getRow(idx));
                if (topK.size() < k)  topK.push({ currentDist, this->labels_.get(idx) });
                else {
                    if (currentDist < topK.top().first) {
                        topK.pop();
                        topK.push({ currentDist, this->labels_.get(idx) });
                    }
                }
            }
        } else {
            double leftDistance = node->left != nullptr
                ? this->distanceEquation_.calculate(target, node->left->centroid)
                : std::numeric_limits<double>::max();
            
            double rightDistance = node->right != nullptr
                ? this->distanceEquation_.calculate(target, node->right->centroid)
                : std::numeric_limits<double>::max();

            double furthestDistance = topK.empty() ? std::numeric_limits<double>::max() 
                : topK.top().first;

            if (leftDistance < rightDistance) {
                if (topK.size() < k)  search(node->left, target, k, topK);
                else if (leftDistance < furthestDistance) search(node->left, target, k, topK);

                if (topK.size() < k) search(node->right, target, k, topK);
                else if (rightDistance < furthestDistance) search(node->right, target, k, topK);
                
            } else if (rightDistance < furthestDistance) {
                if (topK.size() < k) search(node->right, target, k, topK);
                else if (rightDistance < furthestDistance) (node->right, target, k, topK);

                if (topK.size() < k) search(node->left, target, k, topK);
                else if (leftDistance < furthestDistance) search(node->right, target, k, topK);
            }
        }
    }

public:
    BallTree();

    BallTree(Matrix& dataPoints, Vector& labels) {
        this->numOfDimensions_ = dataPoints.numCols;
        this->size_ = dataPoints.numRows; 
        this->dataPoints_ = dataPoints;
        this->labels_ = labels;
    }

    Vector getKNearest(Vector& dataPoint, int k) {
        // First is distance, second is label
        std::priority_queue<std::pair<double, double>> topK;
    }
};

}
