#include <numeric>
#include <jmll/core/vector.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace jmll::core;

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

double Vector::get(int i) const {
    return this->data_[i];
}

std::vector<double> Vector::getData() { return this->data_; }

std::vector<double> Vector::getDataByIndices(std::vector<int> indices) {
    std::vector<double> result;

    for (int index : indices) { result.push_back(this->get(index)); }

    return result;
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

Vector Vector::operator*(double scalar) {
    Vector result(this->data_);

    for (int i = 0; i < result.numCells; i++) {
        result.set(i, result.get(i) * scalar);
    }

    return result;
}

Vector Vector::operator-=(Vector& other) {
    // TODO: dimension check
    // TODO: check both are col vectors or both are row vectors

    for (int i = 0; i < this->numCells; i++) {
        this->set(i, this->get(i) - other.get(i));
    }

    return *this;
}

Vector Vector::operator-(Vector& rhs) {
    return *this -= rhs;
}

Vector Vector::operator+=(const Vector& other) {
    for (int i = 0; i < this->numCells; i++) {
        this->set(i, this->get(i) + other.get(i));
    }

    return *this;
}

Vector Vector::operator+(const Vector& other) {
    return *this += other;
}

Vector Vector::operator/(double scalar) {
    Vector result(this->data_);

    for (int i = 0; i < this->numCells; i++) {
        result.set(i, result.get(i) / scalar);
    }

    return result;
}

double Vector::mean() {
    return std::accumulate(this->data_.begin(), this->data_.end(), 0) / this->numCells;
}

double Vector::median() {
    std::vector<double> dataCopy = this->data_;
    int medianIndex = dataCopy.size() % 2 == 0 ? dataCopy.size() / 2 : dataCopy.size() / 2 + 1;
    auto medianIndexLoc = dataCopy.begin() + medianIndex;
    std::nth_element(dataCopy.begin(), medianIndexLoc, dataCopy.end());
    return dataCopy[medianIndex];
}

double Vector::variance() {
    double mean = this->mean();
    double sum = 0.0;
    for (int i = 0; i < this->numCells; i++) {
        sum += std::pow(this->get(i) - mean, 2.0);
    }

    return sum / (this->numCells - 1);
}
