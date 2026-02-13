#include <mlfs/core/matrix.hpp>
#include <mlfs/core/vector.hpp>
#include <vector>

using namespace mlfs::core;

// Matrix Methods
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

Matrix Matrix::transpose() {
    Matrix res(this->numCols_, this->numRows_);

    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            res.set(c, r, this->get(r, c));
        }
    }

    return res;
}

// Source: https://www.geeksforgeeks.org/computer-science-fundamentals/
// finding-inverse-of-a-matrix-using-gauss-jordan-method/
Matrix Matrix::inverse() {
    int order = this->numRows_;
    double temp;

    // Create copy of matrix
    Matrix mat(this->data_);

    // Create augmented matrix
    for (int i = 0; i < order; i++) {
        for (int j = 0; j < 2 * order; j++) {
            if (j == (i + order)) mat.set(i, j, 1);
        }
    }

    // Interchange the row of matrix
    for (int i = order - 1; i > 0; i--) {
        if (mat.get(i - 1, 0) < mat.get(i, 0)) {
            for (int j = 0; j < 2 * order; j++) {
                temp = mat.get(i, j);
                mat.set(i, j, mat.get(i - 1, j));
                mat.set(i - 1, j, temp);
            }
        }
    }

    // Replace a row by sum of itself and a constant multiple of another row of the matrix
    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            if (j != i) {
                temp = mat.get(j, i) / mat.get(i, j);
                for (int k = 0; k < 2 * order; k++) {
                    double newVal = mat.get(j, k) - mat.get(i, k) * temp;
                    mat.set(j, k, newVal);
                }
            }
        }
    }

    // Multiply each row by a non-zero integer
    for (int i = 0; i < order; i++) {
        temp = mat.get(i, i);
        for (int j = 0; j < 2 * order; j++) {
            mat.set(i, j, mat.get(i, j) / temp);
        }
    }

    return mat;
}
