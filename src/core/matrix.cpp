#include <mlfs/core/matrix.hpp>
#include <vector>

using namespace mlfs::core;

Matrix::Matrix(int numRows, int numCols) {
    this->numRows_ = numRows;
    this->numCols_ = numCols;
    this->data_ = std::vector<std::vector<double>>(
        this->numRows_, 
        std::vector<double>(this->numCols_, 0.0)
    );
}

Matrix::Matrix(std::vector<std::vector<double>>& data) {
    this->numRows_ = data.size();
    this->numCols_ = data.front().size();
    this->data_ = data;
}

double Matrix::get(int r, int c) {
    return this->data_[r][c];
}

void Matrix::set(int r, int c, double val) {
    this->data_[r][c] = val;
}
