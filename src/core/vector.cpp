#include <mlfs/core/vector.hpp>
#include <vector>

using namespace mlfs::core;

Vector::Vector(int numCells) {
    this->numCells_ = numCells;
    this->data_ = std::vector<double>(this->numCells_, 0.0);
}

Vector::Vector(std::vector<double> data) {
    this->numCells_ = data.size();
    this->data_ = data;
}

void Vector::transpose() {
    this->isColVector_ = !this->isColVector_;
}
