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
        
        // If number of remaining points is low, make a leaf node and return it
        if (indices.size() <= LEAF_SIZE) {
            currNode->dataIndices = indices;
        } else {
            // Calculate Centroid using means
            currNode->centroid = Vector(this->numOfDimensions_);
            for (int i = 0; i < indices.size(); i++) {
                currNode->centroid += this->dataPoints_.getRow(indices[i]);
            }

            currNode->centroid /= indices.size();

            // Get radius using distance between centroid and furthest point in space
            for (int i = 0; i < indices.size(); i++) {
                int currDistance = this->distanceEquation_.calculate(
                    currNode->centroid, this->dataPoints_.getRow(i)
                );
                currNode->radius = max(currNode->radius, currDistance);
            }

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

public:
    BallTree();

    BallTree(Matrix& dataPoints, Vector& labels) {
        this->numOfDimensions_ = dataPoints.numCols;
        this->size_ = dataPoints.numRows; 
        this->dataPoints_ = dataPoints;
        this->labels_ = labels;
    }
};

}
