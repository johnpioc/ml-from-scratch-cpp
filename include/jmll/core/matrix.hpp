#pragma once

#include <jmll/core/vector.hpp>
#include <vector>

namespace jmll::core {

class Matrix {
   private:
    /* The underlying data inside the matrix */
    std::vector<std::vector<double>> data_;

    size_t numRows_;
    size_t numCols_;

   public:
    /* Initialises a matrix with a given number of rows and columns with each cell set to 0.0 */
    Matrix(int numRows, int numCols);

    /* Initialises a matrix using a given two-dimensional vector */
    explicit Matrix(std::vector<std::vector<double>> data);

    [[nodiscard]] size_t getNumRows() const noexcept;
    [[nodiscard]] size_t getNumCols() const noexcept;

    /* Retrives the value stored at a given row and column number */
    [[nodiscard]] double get(int r, int c) const;

    /* Retrives the row at the given row index */
    [[nodiscard]] Vector getRow(int r) const;

    [[nodiscard]] Matrix getRows(const std::vector<int>& indices) const;

    [[nodiscard]] Vector getCol(int c) const;

    /* Sets the value at a given row and column number */
    void set(int r, int c, double val);

    Matrix& operator+=(const Matrix& rhs);
    Matrix& operator+=(double rhs);

    Matrix& operator-=(const Matrix& rhs);
    Matrix& operator-=(double rhs);

    Matrix& operator*=(const Matrix& rhs);
    Matrix& operator*=(double rhs);

    Matrix& operator/=(double rhs);

    /* Returns the transposed version of this matrix */
    [[nodiscard]] Matrix transpose() const;

    /* Returns the inverse of this matrix */
    [[nodiscard]] Matrix inverse() const;

    /* Augments the matrix to prepend a column of 1.0s at the start */
    [[nodiscard]] Matrix prependOnes() const noexcept;

    [[nodiscard]] Vector getColMeans() const;

    [[nodiscard]] Vector getRowMeans() const;

    [[nodiscard]] Vector getColVariances() const;

    [[nodiscard]] Vector getRowVariances() const;
};

/* Returns identity matrix of given order */
[[nodiscard]] Matrix identity(int order);

[[nodiscard]] Matrix operator+(Matrix lhs, const Matrix& rhs);
[[nodiscard]] Matrix operator+(Matrix lhs, double rhs);
[[nodiscard]] Matrix operator+(double lhs, Matrix rhs);

[[nodiscard]] Matrix operator-(Matrix lhs, const Matrix& rhs);
[[nodiscard]] Matrix operator-(Matrix lhs, double rhs);
[[nodiscard]] Matrix operator-(double lhs, Matrix rhs);

[[nodiscard]] Matrix operator*(const Matrix& lhs, const Matrix& rhs);
[[nodiscard]] Vector operator*(const Matrix& lhs, const Vector& rhs);
[[nodiscard]] Vector operator*(const Vector& lhs, const Matrix& rhs);
[[nodiscard]] Matrix operator*(Matrix lhs, double rhs);
[[nodiscard]] Matrix operator*(double lhs, Matrix rhs);

[[nodiscard]] Matrix operator/(Matrix lhs, double rhs);
[[nodiscard]] Matrix operator/(double lhs, Matrix rhs);

}  // namespace jmll::core
