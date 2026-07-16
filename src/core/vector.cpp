#include <jmll/core/vector.hpp>
#include <vector>

using namespace jmll::core;

Vector::Vector() { this->data_ = std::vector<double>(0); }

Vector::Vector(int numCells) : numCells_(numCells) {
    this->numCells_ = numCells;
    this->data_ = std::vector<double>(this->numCells_, 0.0);
}

Vector::Vector(std::vector<double> data) {
    this->numCells_ = data.size();
    this->data_ = std::move(data);
}

size_t Vector::getNumCells() const noexcept { return this->numCells_; }
bool Vector::isColVector() const noexcept { return this->isColVector_; }

void Vector::set(int i, double val) { this->data_[i] = val; }

double Vector::get(int i) const { return this->data_[i]; }

std::vector<double> Vector::getData() const { return this->data_; }

std::vector<double> Vector::getDataByIndices(const std::vector<int>& indices) const {
    std::vector<double> result;

    for (int index : indices) { result.push_back(this->get(index)); }

    return result;
}

void Vector::transpose() { this->isColVector_ = !this->isColVector_; }

Vector& Vector::operator+=(const Vector& rhs) {
    // TODO: check vectors are same size
    // TODO: check both are row vectors or both column vectors
    for (int i = 0; i < this->getNumCells(); i++) {
        this->set(i, this->get(i) + rhs.get(i));
    }
    return *this;
}

Vector& Vector::operator+=(double rhs) {
    for (int i = 0; i < this->getNumCells(); i++) {
        this->set(i, this->get(i) + rhs);
    }
    return *this;
}

Vector& Vector::operator-=(const Vector& rhs) {
    // TODO: check vectors are same size
    // TODO: check both are row vectors or both column vectors
    for (int i = 0; i < this->getNumCells(); i++) {
        this->set(i, this->get(i) - rhs.get(i));
    }
    return *this;
}

Vector& Vector::operator-=(double rhs) {
    for (int i = 0; i < this->getNumCells(); i++) {
        this->set(i, this->get(i) - rhs);
    }
    return *this;
}

Vector& Vector::operator*=(double rhs) {
    for (int i = 0; i < this->getNumCells(); i++) {
        this->set(i, this->get(i) * rhs);
    }
    return *this;
}

Vector& Vector::operator/=(double rhs) {
    // TODO: check rhs isn't 0
    for (int i = 0; i < this->getNumCells(); i++) {
        this->set(i, this->get(i) / rhs);
    }
    return *this;
}

//================================================================================================
// VECTOR HELPER FUNCTIONS
//================================================================================================

Vector jmll::core::operator+(Vector lhs, const Vector& rhs) {
    lhs += rhs;
    return lhs;
}

Vector jmll::core::operator+(Vector lhs, double rhs) {
    lhs += rhs;
    return lhs;
}

Vector jmll::core::operator-(Vector lhs, const Vector& rhs) {
    lhs -= rhs;
    return lhs;
}

Vector jmll::core::operator-(Vector lhs, double rhs) {
    lhs -= rhs;
    return lhs;
}

Vector jmll::core::operator*(Vector lhs, double rhs) {
    lhs *= rhs;
    return lhs;
}

Vector jmll::core::operator*(double lhs, Vector rhs) {
    rhs *= lhs;
    return rhs;
}

double jmll::core::operator*(const Vector& lhs, const Vector& rhs) {
    // TODO: check vectors are same size
    double result = 0;

    for (int i = 0; i < lhs.getNumCells(); i++) {
        result += lhs.get(i) * rhs.get(i);
    }

    return result;
}

Vector jmll::core::operator/(Vector lhs, double rhs) {
    lhs /= rhs;
    return lhs;
}
