#include <sstream>
#include <iomanip>

#include "Matrix.h"

// Helper Function Declarations
void printSeperator(int numCols, int* colSpaces);

// ===============================================================================================
// MATRIX OPERATIONS
// ===============================================================================================
void Matrix::cofactor(Matrix& mat, Matrix& buf, int p, int q, int n) 
{
    int i = 0, j = 0;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (row != p && col != q) {
                buf.put(i, j++, mat.get(row, col));
                if (j == n - 1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
}

long long Matrix::determinant(Matrix& mat, int n)
{
    if (n == 1) return this->get(0,0);
    long long det = 0;

    Matrix cof(mat.getNumRows(), mat.getNumCols());

    int sign = 1;
    for (int f = 0; f < n; f++) {
        cofactor(mat, cof, 0, f, n);
        det += sign * mat.get(0, f) * determinant(cof, n - 1);
        sign = -sign;
    }

    return det;
}

void Matrix::adjoint(Matrix& mat, Matrix& buf)
{
    int n = mat.getNumRows();
    if (n == 1) {
        buf.put(0, 0, 1);
        return;
    }

    int sign = 1;
    Matrix cof(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cofactor(mat, cof, i, j, n);
            sign = ((i + j) % 2 == 0) ? 1 : -1;
            buf.put(i, j, sign * determinant(cof, n - 1));
        }
    }
}

Matrix::Matrix()
{
    this->numRows_ = 0;
    this->numCols_ = 0;
    this->data_ = std::vector<std::vector<double>>();
}

Matrix::Matrix(size_t numRows, size_t numCols)
{
    this->numRows_ = numRows;
    this->numCols_ = numCols;

    this->data_ = std::vector<std::vector<double>>(this->numRows_,
        std::vector<double>(this->numCols_, 0.0));
}

Matrix::Matrix(size_t numRows, size_t numCols, std::vector<std::vector<double>> values)
{
    this->numRows_ = numRows;
    this->numCols_ = numCols;

    this->data_ = std::vector<std::vector<double>>(this->numRows_,
        std::vector<double>(this->numCols_));
    for (int i = 0; i < this->numCols_; i++) {
        for (int j = 0; j < this->numRows_; j++) {
            this->data_[j][i] = values[j][i];
        }
    }
}

Matrix Matrix::operator*(const Matrix& other)
{
    if (!canMultiply(other)) {
        throw std::invalid_argument("Cannot multiply matrices with different shapes");
    }

    Matrix product = Matrix(this->numRows_, other.numCols_);
    Matrix otherMatrix(other);

    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < other.numCols_; c++) {
            double dotProduct = 0;
            for (int i = 0; i < this->numCols_; i++) {
                dotProduct += this->get(r, i) * otherMatrix.get(i, c);
            }
            product.put(r, c, dotProduct);
        }
    }

    return product;
}

Matrix Matrix::operator*(double scalar)
{
    Matrix product(this->numRows_, this->numCols_);

    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            product.put(r,c, this->get(r,c) * scalar);
        }
    }

    return product;
}

Matrix Matrix::operator+(const Matrix& other)
{
    if (!canAdd(other)) {
        throw std::invalid_argument("Cannot add matrices with different shapes");
    }

    Matrix otherMatrix(other);
    Matrix sum = Matrix(this->numRows_, this->numCols_);

    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            sum.put(r,c, this->get(r,c) + otherMatrix.get(r,c));
        }
    }

    return sum;
}

Matrix Matrix::operator-(const Matrix& other)
{
    if (!canAdd(other)) {
        throw std::invalid_argument("Cannot subtract matrices with different shapes");
    }

    Matrix otherMatrix(other);
    Matrix difference = Matrix(this->numRows_, this->numCols_);

    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            difference.put(r,c, this->get(r,c) - otherMatrix.get(r,c));
        }
    }

    return difference;
}

Matrix Matrix::dot(const Matrix& other)
{
    if (!canAdd(other)) {
        throw std::invalid_argument("Cannot dot product two matrices with different shapes");
    }

    Matrix otherMatrix(other);
    Matrix dotProduct = Matrix(this->numRows_, this->numCols_);

    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            dotProduct.put(r,c, this->get(r,c) * otherMatrix.get(r,c));
        }
    }

    return dotProduct;
}

Matrix Matrix::transpose()
{
    Matrix transposed(this->numCols_, this->numRows_);

    for (int r = 0; r < this->numRows_; r++) {
        for (int c = 0; c < this->numCols_; c++) {
            transposed.put(c,r, this->get(r,c));
        }
    }

    return transposed;
}

bool Matrix::inverse(Matrix& buf)
{
    int n = this->getNumRows();
    Matrix copy(this->getNumRows(), this->getNumCols(), this->data_);
    long long det = determinant(copy, n);

    if (det == 0) {
        return false;
    }

    Matrix adj(n, n);
    adjoint(copy, adj);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            buf.put(i, j, adj.get(i, j) / static_cast<long double>(det));
        }
    }

    return true;
}

bool Matrix::canMultiply(const Matrix& other)
{
    Matrix otherMatrix(other);
    return this->numCols_ == otherMatrix.getNumRows();
}

bool Matrix::canAdd(const Matrix& other)
{
    Matrix otherMatrix(other);
    return (this->numRows_ == otherMatrix.getNumRows())
        && (this->numCols_ == otherMatrix.getNumCols());
}

size_t Matrix::getNumRows() { return this->numRows_; }
size_t Matrix::getNumCols() { return this->numCols_; }

double Matrix::get(int rowIndex, int colIndex)
{
    if (rowIndex < 0 || rowIndex >= this->numRows_) {
        throw std::invalid_argument("Row index is invalid");
    }

    if (colIndex < 0 || colIndex >= this->numCols_) {
        throw std::invalid_argument("Column index is invalid");
    }

    return this->data_[rowIndex][colIndex];
}

Matrix Matrix::getRow(int rowIndex)
{
    Matrix rowVector(1, this->numCols_);
    for (int i = 0; i < this->numCols_; i++) {
        rowVector.put(0, i, this->get(rowIndex, i));
    }

    return rowVector;
}

Matrix Matrix::getCol(int colIndex)
{
    Matrix colVector(this->numRows_, 1);
    for (int i = 0; i < this->numRows_; i++) {
        colVector.put(i, 0, this->get(i, colIndex));
    }

    return colVector;
}

void Matrix::put(int rowIndex, int colIndex, double value)
{
    this->data_[rowIndex][colIndex] = value;
}

void Matrix::print() 
{
    // For each column, find the maximum number of spaces to allocate
    int colSpaces[this->numCols_];
    for (int colIndex = 0; colIndex < this->numCols_; colIndex++) {
        colSpaces[colIndex] = 0;
        for (int rowIndex = 0; rowIndex < this->numRows_; rowIndex++) {
            double cellValue = this->get(rowIndex, colIndex);
            std::ostringstream stringStream;
            stringStream << std::fixed << std::setprecision(2) << cellValue;

            std::string currentCell = stringStream.str();
            colSpaces[colIndex] = std::max(colSpaces[colIndex], (int) currentCell.size() + 2);
        }
    }

    // Generate a two dimensional array of cell strings to print with respect to the maximum
    // column spaces
    std::string cells[this->numCols_][this->numRows_];
    for (int colIndex = 0; colIndex < this->numCols_; colIndex++) {
        for (int rowIndex = 0; rowIndex < this->numRows_; rowIndex++) {
            double cellValue = this->get(rowIndex, colIndex);
            std::ostringstream stringStream;
            stringStream << std::fixed << std::setprecision(2) << cellValue;
            std::string currentCell = stringStream.str();

            // Add left and right padding to current cell
            int leftPaddingSpaces = 
                std::floor((colSpaces[colIndex] - currentCell.size()) / (double) 2);
            int rightPaddingSpaces = 
                std::ceil((colSpaces[colIndex] - currentCell.size()) / (double) 2);
            std::string formattedString(currentCell);
            while (leftPaddingSpaces--) { formattedString = " " + formattedString; }
            while (rightPaddingSpaces--) { formattedString = formattedString + " "; }
            cells[colIndex][rowIndex] = formattedString;
        }
    }

    // Print table
    printSeperator(this->numCols_, colSpaces);
    for (int rowIndex = 0; rowIndex < this->numRows_; rowIndex++) {
        for (int colIndex = 0; colIndex < this->numCols_; colIndex++) {
            std::cout << "|" << cells[colIndex][rowIndex];
        }
        std::cout << "|\n";
        printSeperator(this->numCols_, colSpaces);
    }
    printSeperator(this->numCols_, colSpaces);
}

// ===============================================================================================
// HELPER FUNCTIONS
// ===============================================================================================

/**
 * @brief Helper function for Matrix print() function that prints a seperator line between rows
 * of the matrix to stdout
 * @param numCols the number of columns in the matrix to print
 * @param colSpaces an array of integers that represent the size of each cell for every column
 */
void printSeperator(int numCols, int *colSpaces) 
{
    for (int colIndex = 0; colIndex < numCols; colIndex++) {
        std::cout << "_";
        for (int i = 0; i < colSpaces[colIndex]; i++) { std:: cout << "_"; }
    }
    std::cout << "_\n";
}
