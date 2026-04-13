#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <vector>

using namespace jmll::core;

// ==============================================================================================
// MATRIX METHODS
// ==============================================================================================
Matrix::Matrix(int numRows, int numCols): 
    numRows(numRows),
    numCols(numCols) {
    this->data_ = std::vector<std::vector<double>>(
        this->numRows, 
        std::vector<double>(this->numCols, 0.0)
    );
}

Matrix::Matrix(std::vector<std::vector<double>>& data) {
    this->numRows = data.size();
    this->numCols = data.front().size();
    this->data_ = data;
}

double Matrix::get(int r, int c) {
    return this->data_[r][c];
}

Vector Matrix::getRow(int r) {
    return Vector(this->data_[r]);
}

Vector Matrix::getCol(int c) {
    std::vector<double> colData(this->numRows);
    for (int i = 0; i < this->numRows; i++) {
        colData[i] = this->get(i, c);
    }
    return Vector(colData);
}

Matrix Matrix::getRows(std::vector<int>& indices) {
    std::vector<std::vector<double>> rows;

    for (int index : indices) {
        rows.push_back(this->getRow(index).getData());
    }

    return Matrix(rows);
}

void Matrix::set(int r, int c, double val) {
    this->data_[r][c] = val;
}

Matrix Matrix::operator*(Matrix& other) {
    // TODO: throw exception if this number of cols does not equal other's num of rows
    
    Matrix res(this->numRows, other.numCols);

    for (int ra = 0; ra < this->numRows; ra++) {
        for (int cb = 0; cb < other.numCols; cb++) {
            double val = 0;
            for (int i = 0; i < this->numCols; i++) {
                val += this->get(ra, i) * other.get(i, cb);
            }
            res.set(ra, cb, val);
        }
    }

    return res;
}

Vector Matrix::operator*(Vector& vec) {
    int common = vec.isColVector ? vec.numCells : 1;
    // TODO: throw exception if the number of cols in matrix doesn't equal commmon
    
    Vector res(this->numRows);
    for (int ra = 0; ra < this->numRows; ra++) {
        double val = 0;
        for (int i = 0; i < common; i++) {
            val += this->get(ra, i) * vec.get(i); 
        }
        res.set(ra, val);
    }

    return res;
}

Matrix Matrix::operator*(double scalar) {
    Matrix res(this->data_);

    for (int r = 0; r < res.numRows; r++) {
        for (int c = 0; c < res.numCols; c++) {
            res.set(r, c, res.get(r,c) * scalar);
        }
    }

    return res;
}

Matrix Matrix::operator+(Matrix& other) {
    // TODO: ensure both matrices are the same shape
    Matrix res(this->data_);

    for (int r = 0; r < res.numRows; r++) {
        for (int c = 0; c < res.numCols; c++) {
            res.set(r,c, this->get(r,c) + other.get(r,c));
        }
    }

    return res;
}

Matrix Matrix::transpose() {
    Matrix res(this->numCols, this->numRows);

    for (int r = 0; r < this->numRows; r++) {
        for (int c = 0; c < this->numCols; c++) {
            res.set(c, r, this->get(r, c));
        }
    }

    return res;
}

// Source: https://www.geeksforgeeks.org/computer-science-fundamentals/
// finding-inverse-of-a-matrix-using-gauss-jordan-method/
Matrix Matrix::inverse() {
    int order = this->numRows;
    
    Matrix aug(order, 2 * order);

    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            aug.set(i, j, this->get(i, j));
            if (i == j) aug.set(i, j + order, 1.0);
        }
    };

    for (int i = order - 1; i > 0; i--) {
        if (aug.get(i - 1, 0) < aug.get(i, 0)) {
            for (int k = 0; k < 2 * order; k++) {
                double temp = aug.get(i, k);
                aug.set(i, k, aug.get(i - 1, k));
                aug.set(i - 1, k, temp);
            }
        }
    }

    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            if (j != i) {
                double temp = aug.get(j, i) / aug.get(i, i);
                for (int k = 0; k < 2 * order; k++) {
                    double val = aug.get(j, k) - (aug.get(i, k) * temp);
                    aug.set(j, k, val);
                }
            }
        }
    }

    Matrix inv(order, order);
    for (int i = 0; i < order; i++) {
        double divisor = aug.get(i, i);
        for (int j = 0; j < order; j++) {
            inv.set(i, j, aug.get(i, j + order) / divisor);
        }
    }

    return inv;
}

Matrix Matrix::prependOnes() {
    Matrix augmented(this->numRows, this->numCols + 1);

    for (int r = 0; r < augmented.numRows; r++) {
        for (int c = 0; c < augmented.numCols; c++) {
            augmented.set(r, c, c == 0 ? 1.0 : this->get(r, c - 1));
        }
    }

    return augmented;
}

Vector Matrix::getColMeans() {
    Vector colMeans(this->numCols);

    for (int r = 0; r < this->numRows; r++) {
        colMeans += this->getRow(r);
    }

    return colMeans / (double) this->numCols;
}

Vector Matrix::getRowMeans() {
    Vector rowMeans(this->numRows);

    for (int c = 0; c < this->numCols; c++) {
        rowMeans += this->getCol(c);
    }

    return rowMeans / (double) this->numRows;
}

Vector Matrix::getColVariances() {
    Vector colVariances(this->numCols);
    Vector colMeans = this->getColMeans();

    for (int r = 0; r < this->numRows; r++) {
        for (int c = 0; c < this->numCols; c++) {
            colVariances += std::pow(this->get(r, c) - colVariances.get(c), 2.0);
        }
    }

    return colVariances / (double) (this->numCols - 1);
}

Vector Matrix::getRowVariances() {
    Vector rowVariances(this->numRows);
    Vector rowMeans = this->getRowMeans();

    for (int r = 0; r < this->numRows; r++) {
        for (int c = 0; c < this->numCols; c++) {
                rowVariances += std::pow(this->get(r, c) - rowVariances.get(r), 2.0);
        }
    }

    return rowVariances / (double) (this->numRows - 1);
}

// ==============================================================================================
// MATRIX HELPERS
// ==============================================================================================
Matrix jmll::core::identity(int order) {
    Matrix res(order, order);

    for (int i = 0; i < order; i++) {
        res.set(i, i, 1.0);
    }

    return res;
}

