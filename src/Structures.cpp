#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>

#include "Structures.h"


namespace Structures {
    void printSeperator(int numCols, int* colSpaces);

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

    Matrix::~Matrix()
    {
        for (int i = 0; i < this->numCols_; i++) {
            delete[] this->data_[i];
        }

        delete[] this->data_;
    }

    // ===========================================================================================
    // HELPERS
    // ===========================================================================================
    void printSeperator(int numCols, int *colSpaces) 
    {
        for (int colIndex = 0; colIndex < numCols; colIndex++) {
            std::cout << "_";
            for (int i = 0; i < colSpaces[colIndex]; i++) { std:: cout << "_"; }
        }
        std::cout << "_\n";
    }
}
