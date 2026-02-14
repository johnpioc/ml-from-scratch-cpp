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

double Vector::operator*(Vector& other) {
    // TODO: check that this vector is a column vector and other is a row vector
    /* TODO: check that the number of cols this vector has equals the number of rows other vector
    * has
    */

    double dotProduct = 0.0;

    for (int i = 0; i < this->numCells; i++) {
        dotProduct += this->get(i) * other.get(i);
    }


    return dotProduct;
}

double Vector::operator*(double scalar) {
    Vector result(this->data_);

    for (int i = 0; i < result.numCells; i++) {
        result.set(i, result.get(i) * scalar);
    }

    return result;
}
