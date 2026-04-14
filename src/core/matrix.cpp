#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <vector>

using namespace jmll::core;

// ==============================================================================================
// MATRIX METHODS
// ==============================================================================================
Matrix::Matrix(int numRows, int numCols) : numRows_(numRows), numCols_(numCols) {
    this->data_ =
        std::vector<std::vector<double>>(this->numRows_, std::vector<double>(this->numCols_, 0.0));
}

Matrix::Matrix(std::vector<std::vector<double>> data) {
    this->numRows_ = data.size();
    this->numCols_ = data.front().size();
    this->data_ = std::move(data);
}

size_t Matrix::getNumRows() const noexcept { return this->numRows_; }
size_t Matrix::getNumCols() const noexcept { return this->numCols_; }

double Matrix::get(int r, int c) const { return this->data_[r][c]; }

Vector Matrix::getRow(int r) const { return Vector(this->data_[r]); }

Matrix Matrix::getRows(const std::vector<int>& indices) const {
    std::vector<std::vector<double>> rows;

    for (int index : indices) {
        rows.push_back(this->getRow(index).getData());
    }

    return Matrix(rows);
}

void Matrix::set(int r, int c, double val) { this->data_[r][c] = val; }

Matrix Matrix::operator*(const Matrix& other) const {
    // TODO: throw exception if this number of cols does not equal other's num of rows

    Matrix res(this->numRows_, other.getNumCols());

    for (int ra = 0; ra < this->numRows_; ra++) {
        for (int cb = 0; cb < other.getNumCols(); cb++) {
            double val = 0;
            for (int i = 0; i < this->numCols_; i++) {
                val += this->get(ra, i) * other.get(i, cb);
            }
            res.set(ra, cb, val);
        }
    }

    return res;
}

Vector Matrix::operator*(const Vector& vec) const {
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

Matrix Matrix::operator*(double scalar) const {
    Matrix res(this->data_);

    for (int r = 0; r < res.getNumRows(); r++) {
        for (int c = 0; c < res.getNumCols(); c++) {
            res.set(r, c, res.get(r, c) * scalar);
        }
    }

    return res;
}

Matrix Matrix::operator+(const Matrix& other) const {
    // TODO: ensure both matrices are the same shape
    Matrix res(this->data_);

    for (int r = 0; r < res.getNumRows(); r++) {
        for (int c = 0; c < res.getNumCols(); c++) {
            res.set(r, c, this->get(r, c) + other.get(r, c));
        }
    }

    return res;
}

Matrix Matrix::transpose() const {
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
Matrix Matrix::inverse() const {
    int order = this->numRows_;

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

Matrix Matrix::prependOnes() const noexcept {
    Matrix augmented(this->numRows_, this->numCols_ + 1);

    for (int r = 0; r < augmented.getNumRows(); r++) {
        for (int c = 0; c < augmented.getNumCols(); c++) {
            augmented.set(r, c, c == 0 ? 1.0 : this->get(r, c - 1));
        }
    }

    return augmented;
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
