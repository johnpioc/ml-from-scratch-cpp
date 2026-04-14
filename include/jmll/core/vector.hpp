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
    Vector(int numCells);

    /* Uses a given vector of doubles to initialise a Vector */
    Vector(std::vector<double> data);

    size_t getNumCells() noexcept;
    bool isColVector() noexcept;

    /* Sets a given value at a given position in the vector */
    void set(int i, double val);

    /* Retrives a value at a given position in the vector */
    double get(int i);

    std::vector<double> getData();

    std::vector<double> getDataByIndices(std::vector<int> indices);

    /* Col Vector <=> Row Vector */
    void transpose();

    /* Dot Product */
    double operator*(const Vector& other);

    /* Scalar Multiplication */
    Vector operator*(double scalar);

    /* Vector Subtraction by Assignmnent */
    Vector operator-=(const Vector& other);

    /* Vector Subtraction */
    Vector operator-(const Vector& rhs);
};
}  // namespace jmll::core
