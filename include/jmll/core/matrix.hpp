#pragma once

#include <jmll/core/vector.hpp>
#include <vector>

namespace jmll::core {

class Matrix {
   private:
    /* The underlying data inside the matrix */
    std::vector<std::vector<double>> data_;

    /* Shape of Matrix */
    size_t numRows_;
    size_t numCols_;

   public:
    /* Initialises a matrix with a given number of rows and columns with each cell set to 0.0 */
    Matrix(int numRows, int numCols);

    /* Initialises a matrix using a given two-dimensional vector */
    explicit Matrix(std::vector<std::vector<double>> data);

    /* Returns the number of rows the matrix has */
    [[nodiscard]] size_t getNumRows() const noexcept;

    /* Returns the number of columns the matrix has */
    [[nodiscard]] size_t getNumCols() const noexcept;

    /* Retrives the value stored at a given row and column number */
    [[nodiscard]] double get(int r, int c) const;

    /* Retrieves the row at the given row index and returns a Vector*/
    [[nodiscard]] Vector getRow(int r) const;

    /* Retrieves selected rows in the matrix and returns a new matrix with those rows */
    [[nodiscard]] Matrix getRows(const std::vector<int>& indices) const;

    /* Retrives the column at the given column index and returns a Vector */
    [[nodiscard]] Vector getCol(int c) const;

    /* Sets the value at a given row and column number */
    void set(int r, int c, double val);

    /* Matrix addition */
    Matrix& operator+=(const Matrix& rhs);

    /* Scalar Addition */
    Matrix& operator+=(double rhs);

    /* Matrix Subtraction */
    Matrix& operator-=(const Matrix& rhs);

    /* Scalar Subtraction */
    Matrix& operator-=(double rhs);

    /* Matrix Multiplication */
    Matrix& operator*=(const Matrix& rhs);

    /* Scalar Multiplication */
    Matrix& operator*=(double rhs);

    /* Scalar Division */
    Matrix& operator/=(double rhs);

    /* Returns the transposed version of this matrix */
    [[nodiscard]] Matrix transpose() const;

    /* Returns the inverse of this matrix */
    [[nodiscard]] Matrix inverse() const;

    /* Augments the matrix to prepend a column of 1.0s at the start */
    [[nodiscard]] Matrix prependOnes() const noexcept;

    /* Calculates the mean of each column in the matrix and returns the result as a Vector */
    [[nodiscard]] Vector getColMeans() const;

    /* Calculates the mean of each row in the matrix and returns the result as a Vector */
    [[nodiscard]] Vector getRowMeans() const;

    /* Calculates the variance of each column in the matrix and returns the result as a Vector */
    [[nodiscard]] Vector getColVariances() const;

    /* Calculates the variance of each row in the matrix and returns the result as a Vector */
    [[nodiscard]] Vector getRowVariances() const;
};

/* Returns identity matrix of given order */
[[nodiscard]] Matrix identity(int order);

[[nodiscard]] Matrix operator+(Matrix lhs, const Matrix& rhs);
[[nodiscard]] Matrix operator+(Matrix lhs, double rhs);

[[nodiscard]] Matrix operator-(Matrix lhs, const Matrix& rhs);
[[nodiscard]] Matrix operator-(Matrix lhs, double rhs);

[[nodiscard]] Matrix operator*(const Matrix& lhs, const Matrix& rhs);
[[nodiscard]] Vector operator*(const Matrix& lhs, const Vector& rhs);
[[nodiscard]] Vector operator*(const Vector& lhs, const Matrix& rhs);
[[nodiscard]] Matrix operator*(Matrix lhs, double rhs);
[[nodiscard]] Matrix operator*(double lhs, Matrix rhs);

[[nodiscard]] Matrix operator/(Matrix lhs, double rhs);

}  // namespace jmll::core
