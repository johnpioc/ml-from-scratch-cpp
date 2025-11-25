#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <random>

#include "Structures.h"

namespace Structures {
    // HELPER FUNCTION DECLARATIONS
    void printSeperator(int numCols, int* colSpaces);
    void relu(Matrix src, Matrix* dest);
    void reluDerivative(Matrix src, Matrix* dest);
    void softmax(Matrix src, Matrix* dest);
    void collapseCols(Matrix src, Matrix* dest);

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

    bool Matrix::canMultiply(const Matrix& other)
    {
        Matrix otherMatrix(other);
        return this->numRows_ == otherMatrix.getNumCols();
    }

    bool Matrix::canAdd(const Matrix& other)
    {
        Matrix otherMatrix(other);
        return (this->numRows_ == otherMatrix.getNumRows())
            && (this->numCols_ == otherMatrix.getNumCols());
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
    // LAYER METHODS
    // ===========================================================================================
    Layer::Layer(size_t neuronsSize, size_t prevLayerNeuronsSize)
    {
        this->neuronsSize_ = neuronsSize;
        this->prevLayerNeuronsSize_ = prevLayerNeuronsSize;

        this->neuronValues_ = Matrix(BATCH_SIZE, this->neuronsSize_);
        this->weights_ = Matrix(this->neuronsSize_, this->prevLayerNeuronsSize_);
        this->bias_ = Matrix(1, this->neuronsSize_);
    }

    Matrix Layer::getNeuronValues() { return this->neuronValues_; };

    void Layer::setNeuronValues(Matrix buf)
    {
        for (int r = 0; r < BATCH_SIZE; r++) {
            for (int c = 0; c < this->neuronsSize_; c++) {
                this->neuronValues_.put(r, c, buf.get(r,c));
            }
        }
    }

    size_t Layer::getNeuronsSize() { return this->neuronsSize_; }
    size_t Layer::getPrevLayerNeuronsSize() { return this->prevLayerNeuronsSize_; }
    Matrix Layer::getWeights() { return this->weights_; }

    void Layer::adjustWeights(Matrix delta)
    {
        this->weights_ = this->weights_ - (delta * LEARNING_RATE);
    }

    void Layer::adjustBias(Matrix delta)
    {
        this->bias_ = this->bias_ - (delta * LEARNING_RATE);
    }

    void Layer::computeNeuronValues(Matrix previousLayerNeuronValues)
    {
        Matrix biasMatrix(BATCH_SIZE, this->neuronsSize_);
        for (int r = 0; r < BATCH_SIZE; r++) {
            for (int c = 0; c < this->neuronsSize_; c++) {
                biasMatrix.put(r, c, this->bias_.get(0, c));
            }
        }

        Matrix neuronValues = (this->weights_ * previousLayerNeuronValues) + biasMatrix;
        relu(neuronValues, &this->neuronValues_);
    }

    // ===========================================================================================
    // NEURAL NETWORK METHODS
    // ===========================================================================================
    void NeuralNetwork::initialiseParams()
    {
    }

    void NeuralNetwork::backPropagate(Matrix expected)
    {
        Layer* outputLayer = this->layers_[this->numLayers_ - 1];
        Matrix softmaxedOutput(outputLayer->getNeuronsSize(),
            outputLayer->getPrevLayerNeuronsSize());
        softmax(outputLayer->getNeuronValues(), &softmaxedOutput);

        Matrix dz = softmaxedOutput - expected;

        for (int layerIndex = this->numLayers_ - 1; layerIndex > 0; layerIndex--) {
            Layer* prevLayer = this->layers_[layerIndex - 1];
            Matrix prevLayerNeuronValues = prevLayer->getNeuronValues();
            Matrix dw = (dz * ((double) 1 / BATCH_SIZE)) * prevLayerNeuronValues;

            Matrix collapsedDz(dz.getNumRows(), dz.getNumCols());
            collapseCols(dz, &collapsedDz);
            Matrix db = collapsedDz * ((double) 1 / BATCH_SIZE);

            Layer* currentLayer = this->layers_[layerIndex];
            currentLayer->adjustWeights(dw);
            currentLayer->adjustBias(db);

            Matrix activationDerivative(dz.getNumCols(), dz.getNumRows());
            reluDerivative(dz, &activationDerivative);
            Matrix newDz = currentLayer->getWeights().transpose() * dz * activationDerivative;
            dz = newDz;
        }
    }

    NeuralNetwork::NeuralNetwork(size_t numLayers, size_t* layerSizes)
    {
        this->numLayers_ = numLayers;
        this->layerSizes_ = new size_t[this->numLayers_];
        this->layers_ = new Layer*[this->numLayers_];
        for (int i = 0; i < this->numLayers_; i++) {
            this->layerSizes_[i] = layerSizes[i];
            this->layers_[i] = 
                new Layer(i == 0 ? BATCH_SIZE : layerSizes[i - 1], layerSizes[i]);
        }
        this->initialiseParams();
    }

    void NeuralNetwork::feedForward(Matrix batch)
    {
        Layer* inputLayer = this->layers_[0];
        inputLayer->setNeuronValues(batch);

        for (int layerIndex = 1; layerIndex < this->numLayers_; layerIndex++) {
            Layer* currentLayer = this->getLayer(layerIndex);
            Layer* previousLayer = this->getLayer(layerIndex - 1);
            currentLayer->computeNeuronValues(previousLayer->getNeuronValues());
        }
    }

    Layer* NeuralNetwork::getLayer(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= this->numLayers_)
            throw std::invalid_argument("Layer index is invalid");

        return this->layers_[layerIndex];
    }

    void NeuralNetwork::train(std::vector<Matrix> observations, std::vector<Matrix> expected)
    {
        int numOfBatches = observations.size();

        for (int batchIndex = 0; batchIndex < numOfBatches; batchIndex++) {
            Matrix currentBatch = observations[batchIndex];
            Matrix currentExpectedMatrix = expected[batchIndex];

            this->feedForward(currentBatch);
            this->backPropagate(currentExpectedMatrix);
        }
    }

    NeuralNetwork::~NeuralNetwork()
    {
        for (int i = 0; i < this->numLayers_; i++) {
            delete this->layers_[i];
        }
        delete[] this->layers_;
        delete[] layerSizes_;
    }

    // ===========================================================================================
    // HELPERS
    // ===========================================================================================
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

    /** Rectified linear unit (squishification function for neuron values) */
    void relu(Matrix src, Matrix* dest)
    {
        for (int r = 0; r < src.getNumRows(); r++) {
            for (int c = 0; c < src.getNumCols(); c++) {
                dest->put(r, c, src.get(r, c) > 0 ? src.get(r, c) : 0);
            }
        }
    }

    /** Derivative of the relu */
    void reluDerivative(Matrix src, Matrix* dest)
    {
        for (int r = 0; r < src.getNumRows(); r++) {
            for (int c = 0; c < src.getNumCols(); c++) {
                dest->put(r, c, src.get(r, c) > 0 ? 1 : 0);
            }
        }
    }

    /** Applies the softmax function to a given matrix */
    void softmax(Matrix src, Matrix* dest)
    {
        for (int r = 0; r < src.getNumRows(); r++) {
            double denom = 0;
            for (int c = 0; c < src.getNumCols(); c++) {
                denom += std::exp(src.get(r,c));
            }

            for (int c = 0; c < src.getNumCols(); c++) {
                dest->put(r,c, std::exp(src.get(r,c)) / denom);
            }
        }
    }

    /** Sums up all row vectors in a single number and outputs a (r x 1) matrix */
    void collapseCols(Matrix src, Matrix* dest)
    {
        for (int r = 0; r < src.getNumRows(); r++) {
            double sum = 0;
            for (int c = 0; c < src.getNumCols(); c++) {
                sum += src.get(r,c);
            }

            dest->put(r, 0, sum);
        }
    }
}
