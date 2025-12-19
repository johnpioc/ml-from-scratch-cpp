#pragma once

#include <iostream>
#include <cstdint>
#include <vector>

struct Matrix {
private:
    // The number of rows in the matrix
    size_t numRows_;
    // The number of columns in the matrix
    size_t numCols_;
    // The underlying data inside the matrix
    std::vector<std::vector<double>> data_;

    void cofactor(Matrix& mat, Matrix& buf, int p, int q, int n);
    void adjoint(Matrix& mat, Matrix& buf);

public:
    Matrix();

    /**
     * @brief constructs a new matrix struct and initialises all cell values to 0.0
     * @param numRows the number of rows in the matrix
     * @param numCols the number of columns in the matrix
     */
    Matrix(size_t numRows, size_t numCols);

    /**
     * @brief constructs a new matrix struct 
     * @param numRows the number of rows in the matrix
     * @param numCols the number of columns in the matrix
     * @param values the cell values inside the matrix
     */
    Matrix(size_t numRows, size_t numCols, std::vector<std::vector<double>> values);

    // Operator Overloads
    Matrix operator*(const Matrix& other);
    Matrix operator*(const double scalar);
    Matrix operator+(const Matrix& other);
    Matrix operator-(const Matrix& other);

    Matrix dot(const Matrix& other);

    /**
     * @brief constructs a transposed version of this matrix
     * @returns a new matrix object that is the tranposed version
     */
    Matrix transpose();

    long long determinant(Matrix& mat, int n);
    bool inverse(Matrix& buf);

    /**
     * Returns true if this matrix and given matrix can be multiplied, otherwise false
     */
    bool canMultiply(const Matrix& other);

    /**
     * Returns true if this matrix and given matrix can be added (or subtracted), otherwise
     * false
     */
    bool canAdd(const Matrix& other);

    /** Returns the number of rows inside the matrix */
    size_t getNumRows();
    /** Returns the number of columns inside the matrix */
    size_t getNumCols();

    /**
     * @brief Returns a cell value 
     * @param rowIndex the index of the row of the cell we want to get the value from 
     * @param colIndex the index of the column of the cell we want to get the value from
     * @return the cell value corresponding to the row and column index
     */
    double get(int rowIndex, int colIndex);

    Matrix getRow(int rowIndex);
    Matrix getCol(int colIndex);

    /**
     * @brief inserts/updates a given cell in the matrix
     * @param rowIndex the index of the row of the cell we want to update
     * @param colIndex the index of the column of the cell we want to update
     * @param value the value we want to store in the cell
     */
    void put(int rowIndex, int colIndex, double value);

    /** Prints the contents of the matrix to stdout */
    void print();
};
