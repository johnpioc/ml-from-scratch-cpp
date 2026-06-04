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

size_t Vector::getNumCells() const noexcept { return this->numCells_; }
bool Vector::isColVector() const noexcept { return this->isColVector_; }

void Vector::set(int i, double val) { this->data_[i] = val; }

double Vector::get(int i) const { return this->data_[i]; }

std::vector<double> Vector::getData() const { return this->data_; }

std::vector<double> Vector::getDataByIndices(const std::vector<int>& indices) const {
    std::vector<double> result;

    for (int index : indices) {
        result.push_back(this->get(index));
    }

    return result;
}

void Vector::transpose() { this->isColVector_ = !this->isColVector_; }

//================================================================================================
// VECTOR HELPER FUNCTIONS
//================================================================================================

Vector operator+(Vector lhs, const Vector& rhs) {
    lhs += rhs;
    return lhs;
}

Vector operator+(Vector lhs, double rhs) {
    lhs += rhs;
    return lhs;
}

Vector operator-(Vector lhs, const Vector& rhs) {
    lhs -= rhs;
    return lhs;
}

Vector operator-(Vector lhs, double rhs) {
    lhs -= rhs;
    return lhs;
}

Vector operator*(Vector lhs, double rhs) {
    lhs *= rhs;
    return lhs;
}

Vector operator*(double lhs, Vector rhs) {
    rhs *= lhs;
    return rhs;
}

Vector operator*(Vector lhs, const Vector& rhs) {
    lhs *= rhs;
    return lhs;
}

Vector operator/(Vector lhs, double rhs) {
    lhs /= rhs;
    return lhs;
}
