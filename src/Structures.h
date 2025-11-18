#pragma once

#include <iostream>
#include <cstdint>

namespace Structures {
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
        void print();

        ~Matrix();
    };
}
