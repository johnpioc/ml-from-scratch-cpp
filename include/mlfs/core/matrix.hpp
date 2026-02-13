#pragma once

#include <mlfs/core/vector.hpp>
#include <vector>

namespace mlfs {
namespace core {

class Matrix {
private:
    /* The number of rows in the matrix */
    int numRows_;

    /* The number of columns in the matrix */
    int numCols_;

    /* The underlying data inside the matrix */
    std::vector<std::vector<double>> data_;

public:
    /* Initialises a matrix with a given number of rows and columns with each cell set to 0.0 */
    Matrix(int numRows, int numCols);

    /* Initialises a matrix using a given two-dimensional vector */
    Matrix(std::vector<std::vector<double>>& data);

    /* Retrieves the number of rows this matrix has */
    int getNumRows();

    /* Retrives the number of cols this matrix has */
    int getNumCols();

    /* Retrives the value stored at a given row and column number */
    double get(int r, int c);

    /* Sets the value at a given row and column number */
    void set(int r, int c, double val);

    /* Matrix multiplication operator overload */
    Matrix operator*(Matrix& other);

    /* Matrix multiplication with vector operator overload */
    Vector operator*(Vector& vec);

    /* Returns the transposed version of this matrix */
    Matrix transpose();

    /* Returns the inverse of this matrix */
    Matrix inverse();
};

}
}
