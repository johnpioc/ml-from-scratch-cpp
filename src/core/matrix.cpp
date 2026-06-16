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

Vector Matrix::getCol(int c) const {
    // TODO: check c is within bounds
    Vector result(this->getNumRows());

    for (int i = 0; i < result.getNumCells(); i++) {
        result.set(i, this->get(i, c));
    }
    return result;
}

void Matrix::set(int r, int c, double val) { this->data_[r][c] = val; }

Matrix& Matrix::operator+=(const Matrix& rhs) {
    // TODO: check both matrices are same shape
    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            this->set(r, c, this->get(r, c) + rhs.get(r, c));
        }
    }

    return *this;
}

Matrix& Matrix::operator+=(double rhs) {
    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            this->set(r, c, this->get(r, c) + rhs);
        }
    }

    return *this;
}

Matrix& Matrix::operator-=(const Matrix& rhs) {
    // TODO: check both matrices are same shape
    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            this->set(r, c, this->get(r, c) - rhs.get(r, c));
        }
    }

    return *this;
}

Matrix& Matrix::operator-=(double rhs) {
    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            this->set(r, c, this->get(r, c) - rhs);
        }
    }

    return *this;
}

Matrix& Matrix::operator*=(const Matrix& rhs) {
    *this = *this * rhs;
    return *this;
}

Matrix& Matrix::operator*=(double rhs) {
    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            this->set(r, c, this->get(r, c) * rhs);
        }
    }

    return *this;
}

Matrix& Matrix::operator/=(double rhs) { return *this *= (1.0 / rhs); }

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

Vector Matrix::getColMeans() const {
    Vector colMeans(this->numCols_);
    colMeans.transpose();

    for (int r = 0; r < this->numRows_; r++) {
        colMeans += this->getRow(r);
    }

    return colMeans / static_cast<double>(this->numRows_);
}

Vector Matrix::getRowMeans() const {
    Vector rowMeans(this->numRows_);

    for (int c = 0; c < this->numCols_; c++) {
        rowMeans += this->getCol(c);
    }

    return rowMeans / static_cast<double>(this->numCols_);
}

Vector Matrix::getColVariances() const {
    Vector colVariances(this->numCols_);
    Vector colMeans = this->getColMeans();
    colVariances.transpose();

    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            double residual = this->get(r, c) - colMeans.get(c);
            colVariances.set(c, colVariances.get(c) + residual * residual);
        }
    }

    return colVariances / static_cast<double>(this->numRows_ - 1);
}

Vector Matrix::getRowVariances() const {
    Vector rowVariances(this->numRows_);
    Vector rowMeans = this->getRowMeans();

    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            double residual = this->get(r, c) - rowMeans.get(r);
            rowVariances.set(r, rowVariances.get(r) + residual * residual);
        }
    }

    return rowVariances / static_cast<double>(this->numCols_ - 1);
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

Matrix jmll::core::operator+(Matrix lhs, const Matrix& rhs) {
    lhs += rhs;
    return lhs;
}

Matrix jmll::core::operator+(Matrix lhs, double rhs) {
    lhs += rhs;
    return lhs;
}

Matrix jmll::core::operator-(Matrix lhs, const Matrix& rhs) {
    lhs -= rhs;
    return lhs;
}

Matrix jmll::core::operator-(Matrix lhs, double rhs) {
    lhs -= rhs;
    return lhs;
}

Matrix jmll::core::operator*(const Matrix& lhs, const Matrix& rhs) {
    // TODO: throw exception if lhs columns != rhs rows
    int m = lhs.getNumRows();
    int n = lhs.getNumCols();
    int p = rhs.getNumCols();

    Matrix result(m, p);

    for (int ra = 0; ra < m; ra++) {
        for (int cb = 0; cb < p; cb++) {
            double val = 0.0;
            for (int i = 0; i < n; i++) {
                val += lhs.get(ra, i) * rhs.get(i, cb);
            }
            result.set(ra, cb, val);
        }
    }
    return result;
}

Vector jmll::core::operator*(const Matrix& lhs, const Vector& rhs) {
    // TODO: check lhs cols == rhs rows
    // TODO: check rhs is a column vector

    int common = lhs.getNumCols();
    Vector result(lhs.getNumRows());

    for (int r = 0; r < lhs.getNumRows(); r++) {
        double value = 0.0;
        for (int i = 0; i < common; i++) {
            value += lhs.get(r, i) * rhs.get(i);
        }
        result.set(r, value);
    }

    return result;
}

Vector jmll::core::operator*(const Vector& lhs, const Matrix& rhs) {
    // TODO: check lhs is a row vector
    // TODO: check lhs cols = rhs rows

    int common = lhs.getNumCells();
    Vector result(rhs.getNumCols());
    result.transpose();

    for (int c = 0; c < rhs.getNumCols(); c++) {
        double value = 0.0;
        for (int i = 0; i < common; i++) {
            value += lhs.get(i) * rhs.get(i, c);
        }
        result.set(c, value);
    }

    return result;
}

Matrix jmll::core::operator*(Matrix lhs, double rhs) {
    lhs *= rhs;
    return lhs;
}

Matrix jmll::core::operator*(double lhs, Matrix rhs) {
    rhs *= lhs;
    return rhs;
}

Matrix jmll::core::operator/(Matrix lhs, double rhs) {
    // Check that rhs != 0
    lhs /= rhs;
    return lhs;
}
