#pragma once

#include <iostream>
#include <cstdint>

namespace Structures {

    /**
    * Column Vector
    */
    struct Vector {
    private:
        size_t numRows_;
        double* data_;

    public:
        Vector(size_t numRows);
        Vector(size_t numRows, double* values);

        size_t getNumRows();
        double* getData();
        double get(int index);
        void put(int index, double value);

        Vector& operator+(const Vector& vector);
        Vector& operator+(const double value);
        Vector& operator-(const Vector& vector);
        Vector& operator-(const double value);
        Vector& operator*(const Vector& vector);
        Vector& operator*(const double value);

        double dot(const Vector& other);
    };

    struct Matrix {
    private:
        size_t numRows_;
        size_t numCols_;
        double** data_;

    public:
        Matrix(size_t numRows, size_t numCols);
        Matrix(size_t numRows, size_t numCols, double** values);
        
        size_t getNumRows();
        size_t getNumCols();
        double** getData();
        double get(int rowIndex, int colIndex);
        void put(int rowIndex, int colIndex, double value);
    }
}
