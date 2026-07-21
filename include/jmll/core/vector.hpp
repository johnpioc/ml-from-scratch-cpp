#pragma once

#include <cstddef>
#include <vector>

namespace jmll::core {

class Vector {
   private:
    /* The underlying data inside the vector */
    std::vector<double> data_;

    /* Indicator if vector is a column vector */
    bool isColVector_ = true;

    /* Size of the vector */
    size_t numCells_;

   public:
    /* Initialises an empty vector with no cells */
    explicit Vector();

    /* Initialises a vector with a given number of cells to 0.0 */
    explicit Vector(int numCells);

    /* Uses a given vector of doubles to initialise a Vector */
    explicit Vector(std::vector<double> data);

    /* Returns the size of the vector */
    [[nodiscard]] size_t getNumCells() const noexcept;

    /* Returns true if vector is a column vector, false otherwise */
    [[nodiscard]] bool isColVector() const noexcept;

    /* Sets a given value at a given position in the vector */
    void set(int i, double val);

    /* Retrives a value at a given position in the vector */
    [[nodiscard]] double get(int i) const;

    /* Returns a std::vector of the Vector's underlying data */
    [[nodiscard]] std::vector<double> getData() const;

    /* Returns a std::vector of Vector's underlying data w/ respect to given indices */
    [[nodiscard]] std::vector<double> getDataByIndices(const std::vector<int>& indices) const;

    /* Col Vector <=> Row Vector */
    void transpose();

    /* Vector Addition */
    Vector& operator+=(const Vector& rhs);

    /* Scalar Addition */
    Vector& operator+=(double rhs);

    /* Vector Subtraction */
    Vector& operator-=(const Vector& rhs);

    /* Scalar Subtraction */
    Vector& operator-=(double rhs);

    /* Scalar Multiplication */
    Vector& operator*=(double rhs);

    /* Vector Element-wise Division */
    Vector& operator/=(double rhs);

    /* Returns the sum of all the elements in the vector */
    [[nodiscard]] double getSum() const;
};

[[nodiscard]] Vector operator+(Vector lhs, const Vector& rhs);
[[nodiscard]] Vector operator+(Vector lhs, double rhs);

[[nodiscard]] Vector operator-(Vector lhs, const Vector& rhs);
[[nodiscard]] Vector operator-(Vector lhs, double rhs);

[[nodiscard]] Vector operator*(Vector lhs, double rhs);
[[nodiscard]] Vector operator*(double lhs, Vector rhs);

// Dot product
[[nodiscard]] double operator*(const Vector& lhs, const Vector& rhs);

[[nodiscard]] Vector operator/(Vector lhs, double rhs);
}  // namespace jmll::core
