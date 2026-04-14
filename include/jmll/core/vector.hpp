#pragma once

#include <cstddef>
#include <vector>

namespace jmll::core {

class Vector {
   private:
    /* The underlying data inside the vector */
    std::vector<double> data_;

    bool isColVector_ = true;
    size_t numCells_;

   public:
    /* Initialises a vector with a given number of cells to 0.0 */
    explicit Vector(int numCells);

    /* Uses a given vector of doubles to initialise a Vector */
    explicit Vector(std::vector<double> data);

    [[nodiscard]] size_t getNumCells() const noexcept;
    [[nodiscard]] bool isColVector() const noexcept;

    /* Sets a given value at a given position in the vector */
    void set(int i, double val);

    /* Retrives a value at a given position in the vector */
    [[nodiscard]] double get(int i) const;

    [[nodiscard]] std::vector<double> getData() const;

    [[nodiscard]] std::vector<double> getDataByIndices(const std::vector<int>& indices) const;

    /* Col Vector <=> Row Vector */
    void transpose();

    /* Dot Product */
    [[nodiscard]] double operator*(const Vector& other) const;

    /* Scalar Multiplication */
    [[nodiscard]] Vector operator*(double scalar) const;

    /* Vector Subtraction by Assignmnent */
    [[nodiscard]] Vector operator-=(const Vector& other);

    /* Vector Subtraction */
    [[nodiscard]] Vector operator-(const Vector& rhs);
};
}  // namespace jmll::core
