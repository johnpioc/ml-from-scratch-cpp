#include <jmll/core/vector.hpp>
#include <vector>

using namespace jmll::core;

Vector::Vector(int numCells) : numCells_(numCells) {
    this->numCells_ = numCells;
    this->data_ = std::vector<double>(this->numCells_, 0.0);
}

Vector::Vector(std::vector<double> data) {
    this->numCells_ = data.size();
    this->data_ = std::move(data);
}

size_t Vector::getNumCells() noexcept { return this->numCells_; }
bool Vector::isColVector() noexcept { return this->isColVector_; }

void Vector::set(int i, double val) { this->data_[i] = val; }

double Vector::get(int i) { return this->data_[i]; }

std::vector<double> Vector::getData() { return this->data_; }

std::vector<double> Vector::getDataByIndices(const std::vector<int> indices) {
    std::vector<double> result;

    for (int index : indices) {
        result.push_back(this->get(index));
    }

    return result;
}

void Vector::transpose() { this->isColVector_ = !this->isColVector_; }

double Vector::operator*(const Vector& other) {
    // TODO: check that this vector is a column vector and other is a row vector
    /* TODO: check that the number of cols this vector has equals the number of rows other vector
     * has
     */

    double dotProduct = 0.0;

    for (int i = 0; i < this->numCells_; i++) {
        dotProduct += this->get(i) * other.get(i);
    }

    return dotProduct;
}

Vector Vector::operator*(double scalar) {
    Vector result(this->data_);

    for (int i = 0; i < result.getNumCells(); i++) {
        result.set(i, result.get(i) * scalar);
    }

    return result;
}

Vector Vector::operator-=(const Vector& other) {
    // TODO: dimension check
    // TODO: check both are col vectors or both are row vectors

    for (int i = 0; i < this->numCells_; i++) {
        this->set(i, this->get(i) - other.get(i));
    }

    return *this;
}

Vector Vector::operator-(const Vector& rhs) { return *this -= rhs; }
