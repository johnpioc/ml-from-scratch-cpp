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

int Vector::getNumCells() const { return this->numCells_; }

bool Vector::isColVector() { return this->isColVector_; }

void Vector::set(int i, double val) {
    this->data_[i] = val;
}

double Vector::get(int i) {
    return this->data_[i];
}

void Vector::transpose() {
    this->isColVector_ = !this->isColVector_;
}
