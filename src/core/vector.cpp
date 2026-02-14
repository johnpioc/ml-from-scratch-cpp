#include <mlfs/core/vector.hpp>
#include <vector>

using namespace mlfs::core;

Vector::Vector(int numCells):
    numCells(numCells) {
    this->numCells = numCells;
    this->data_ = std::vector<double>(this->numCells, 0.0);
}

Vector::Vector(std::vector<double> data) {
    this->numCells = data.size();
    this->data_ = data;
}

void Vector::set(int i, double val) {
    this->data_[i] = val;
}

double Vector::get(int i) {
    return this->data_[i];
}

void Vector::transpose() {
    this->isColVector = !this->isColVector;
}
