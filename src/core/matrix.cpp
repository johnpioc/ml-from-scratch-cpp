#include <mlfs/core/matrix.hpp>
#include <mlfs/core/vector.hpp>
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

Matrix Matrix::operator*(Matrix& other) {
    // TODO: throw exception if this number of cols does not equal other's num of rows
    
    Matrix res(this->numRows_, other.numCols_);

    for (int ra = 0; ra < this->numRows_; ra++) {
        for (int cb = 0; cb < other.numCols_; cb++) {
            double val = 0;
            for (int i = 0; i < this->numCols_; i++) {
                val += this->get(ra, i) * this->get(i, cb);
            }
            res.set(ra, cb, val);
        }
    }

    return res;
}

Vector Matrix::operator*(Vector& vec) {
    int common = vec.isColVector() ? vec.getNumCells() : 1;
    // TODO: throw exception if the number of cols in matrix doesn't equal commmon
    
    Vector res(this->numRows_);
    for (int ra = 0; ra < this->numRows_; ra++) {
        double val = 0;
        for (int i = 0; i < common; i++) {
            val += this->get(ra, i) * vec.get(i); 
        }
        res.set(ra, val);
    }

    return res;
}
