#include "Structures.h"
#include <stdexcept>

namespace Structures {
    // ===========================================================================================
    // VECTOR METHODS 
    // ===========================================================================================
    Vector::Vector(size_t numRows)
    {
        if (numRows <= 0) {
            throw std::invalid_argument("number of rows cannot be less than or equal to zero");
        }

        this->data_ = new double[numRows]();
        this->numRows_ = numRows;
    }

    Vector::Vector(size_t numRows, double *values)
    {
        if (numRows <= 0) {
            throw std::invalid_argument("number of rows cannot be less than or equal to zero");
        }

        if (values == NULL) {
            throw std::invalid_argument("values array cannot be empty");
        }

        this->data_ = new double[numRows]();
        this->numRows_ = numRows;

        for (int i = 0; i < numRows; i++) {
            this->data_[i] = values[i];
        }
    }

    size_t Vector::getNumRows()
    {
        return this->numRows_;
    }

    double* Vector::getData()
    {
        return this->data_;
    }
    
    double Vector::get(int index)
    {
        if (index < 0 || index >= this->numRows_) {
            throw std::invalid_argument("index " + std::to_string(index) + 
                " is invalid for length " + std::to_string(this->numRows_));
        }

        return this->data_[index];
    }

    void Vector::put(int index, double value) {
        if (index < 0 || index >= this->numRows_) {
            throw std::invalid_argument("index " + std::to_string(index) + 
                " is invalid for length " + std::to_string(this->numRows_));
        }

        this->data_[index] = value;
    }

    Vector& Vector::operator+(const Vector& vector)
    {
        Vector other(vector);

        if (this->getNumRows() != other.getNumRows()) {
            throw std::invalid_argument("vectors have different shapes");
        }

        double sum[this->numRows_];
        for (int i = 0; i < this->numRows_; i++) {
            sum[i] = this->get(i) + other.get(i);
        }

        Vector newVector(this->numRows_, sum);
        return newVector;
    }

    Vector& Vector::operator+(const double value)
    {
        double sum[this->numRows_];
        for (int i = 0; i < this->numRows_; i++) {
            sum[i] = this->get(i) + value;
        }

        Vector newVector(this->numRows_, sum);
        return newVector;
    }

    Vector& Vector::operator-(const Vector& vector)
    {
        Vector other(vector);

        if (this->getNumRows() != other.getNumRows()) {
            throw std::invalid_argument("vectors have different shapes");
        }
        
        double diff[this->numRows_];
        for (int i = 0; i < this->numRows_; i++) {
            diff[i] = this->get(i) - other.get(i);
        }

        Vector newVector(this->numRows_, diff);
        return newVector;
    }

    Vector& Vector::operator-(const double value)
    {
        double diff[this->numRows_];
        for (int i = 0; i < this->numRows_; i++) {
            diff[i] = this->get(i) - value;
        }

        Vector newVector(this->numRows_, diff);
        return newVector;
    }

    Vector& Vector::operator*(const Vector& vector)
    {
        Vector other(vector);

        if (this->getNumRows() != other.getNumRows()) {
            throw std::invalid_argument("vectors have different shapes");
        }

        double prod[this->numRows_];
        for (int i = 0; i < this->numRows_; i++) {
            prod[i] = this->get(i) * other.get(i);
        }

        Vector newVector(this->numRows_, prod);
        return newVector;
    }

    Vector& Vector::operator*(const double value)
    {
        double prod[this->numRows_];
        for (int i = 0; i < this->numRows_; i++) {
            prod[i] = this->get(i) * value;
        }

        Vector newVector(this->numRows_, prod);
        return newVector;
    }

    double Vector::dot(const Vector& vector) {
        Vector other(vector);

        if (this->getNumRows() != other.getNumRows()) {
            throw std::invalid_argument("vectors have different shapes");
        }

        double result = 0;
        for (int i = 0; i < this->numRows_; i++) {
            result += this->get(i) * other.get(i);
        }

        return result;
    }

    // ===========================================================================================
    // MATRIX METHODS
    // ===========================================================================================
    Matrix::Matrix(size_t numRows, size_t numCols)
    {
        this->numRows_ = numRows;
        this->numCols_ = numCols;

        this->data_ = new double*[this->numCols_];
        for (int i = 0; i < this->numCols_; i++) {
            this->data_[i] = new double[this->numRows_]();
        }
    }

    Matrix::Matrix(size_t numRows, size_t numCols, double** values)
    {
        this->numRows_ = numRows;
        this->numCols_ = numCols;

        this->data_ = new double*[this->numCols_];
        for (int i = 0; i < this->numCols_; i++) {
            this->data_[i] = new double[this->numRows_];
            for (int j = 0; j < this->numRows_; j++) {
                this->data_[i][j] = values[i][j];
            }
        }
    }
    
    size_t Matrix::getNumRows() { return this->numRows_; }
    size_t Matrix::getNumCols() { return this->numCols_; }
    double** Matrix::getData() { return this->data_; }

    double Matrix::get(int rowIndex, int colIndex)
    {
        if (rowIndex < 0 || rowIndex >= this->numRows_) {
            throw std::invalid_argument("Row index is invalid");
        }

        if (colIndex < 0 || colIndex >= this->numCols_) {
            throw std::invalid_argument("Column index is invalid");
        }

        return this->data_[colIndex][rowIndex];
    }

    void Matrix::put(int rowIndex, int colIndex, double value)
    {
        if (rowIndex < 0 || rowIndex >= this->numRows_) {
            throw std::invalid_argument("Row index is invalid");
        }

        if (colIndex < 0 || colIndex >= this->numCols_) {
            throw std::invalid_argument("Column index is invalid");
        }
        
        this->data_[colIndex][rowIndex] = value;
    }
}
