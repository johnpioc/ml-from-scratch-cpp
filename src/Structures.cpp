#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <limits>

#include "Structures.h"


namespace Structures {
    void printSeperator(int numCols, int* colSpaces);
    float relu(double raw);
    double dotProduct(double* vectorA, float* vectorB, int length);

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

    void Matrix::getRow(int rowIndex, double** buf, size_t* bufSize)
    {
        (*buf) = new double[this->numCols_];
        (*bufSize) = this->numCols_;

        for (int colIndex = 0; colIndex < this->numCols_; colIndex++) {
            (*buf)[colIndex] = this->get(rowIndex, colIndex);
        }
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
    // LAYER METHODS
    // ===========================================================================================
    Layer::Layer(size_t neuronsSize, size_t prevLayerNeuronsSize)
    {
        this->neuronsSize_ = neuronsSize;
        this->prevLayerNeuronsSize_ = prevLayerNeuronsSize;

        this->neuronValues_ = new float[this->neuronsSize_];
        this->weights_ = new Matrix(this->neuronsSize_, this->prevLayerNeuronsSize_);
    }

    void Layer::setWeight(int currLayerNeuronIndex, int prevLayerNeuronIndex, double value)
    {
        this->weights_->put(currLayerNeuronIndex, prevLayerNeuronIndex, value);
    }

    double Layer::getWeight(int currLayerNeuronIndex, int prevLayerNeuronIndex)
    {
        return this->weights_->get(currLayerNeuronIndex, prevLayerNeuronIndex);
    }

    void Layer::computeNeuronValues(float** prevLayerNeuronValues)
    {
        for (int i = 0; i < this->neuronsSize_; i++) {
            double* currentWeights;
            size_t bufSize;
            this->weights_->getRow(i, &currentWeights, &bufSize);
            this->neuronValues_[i] = 
                relu(
                    dotProduct(
                        currentWeights, (*prevLayerNeuronValues),
                        this->prevLayerNeuronsSize_
                    ) - this->bias_ 
                );
            delete[] currentWeights;
        }
    }

    void Layer::setNeuronValues(float* values)
    {
        for (int i = 0; i < this->neuronsSize_; i++) {
            this->neuronValues_[i] = values[i];
        }
    }

    float Layer::getNeuronValue(int neuronIndex)
    {
        if (neuronIndex < 0 || neuronIndex >= this->neuronsSize_)
            throw std::invalid_argument("invalid neuron index");

        return this->neuronValues_[neuronIndex];
    }

    void Layer::getNeuronValues(float** buf)
    {
        (*buf) = new float[this->neuronsSize_];
        for (int i = 0; i < this->neuronsSize_; i++) {
            (*buf)[i] = this->getNeuronValue(i);
        }
    }

    void Layer::setBias(double bias) { this->bias_ = bias; }
    double Layer::getBias() { return this->bias_; }

    Layer::~Layer()
    {
        delete this->weights_;
        delete[] neuronValues_;
    }

    // ===========================================================================================
    // NEURAL NETWORK METHODS
    // ===========================================================================================
    void NeuralNetwork::initialiseParams()
    {
        // Apply a weight of 1 and a bias of 0 to the first layer as its the input layer
        Layer* inputLayer = this->layers_[0];
        inputLayer->setBias(0.0);
        for (int i = 0; i < this->layerSizes_[0]; i++) {
            for (int j = 0; j < this->layerSizes_[0]; j++) {
                inputLayer->setWeight(i, j, 1.0);
            }
        }

        double lowerBound = std::numeric_limits<double>::min();
        double upperBound = std::numeric_limits<double>::max();
        std::uniform_real_distribution<double> rand(lowerBound, upperBound);
        std::default_random_engine re;
        re.seed(67);

        // For rest of layers, apply a random weight and bias
        for (int layerIndex = 1; layerIndex < this->numLayers_; layerIndex++) {
            Layer* currentLayer = this->layers_[layerIndex];
            currentLayer->setBias(rand(re));
            for (int currLayerNeuronIndex = 0; currLayerNeuronIndex < 
                this->layerSizes_[layerIndex]; currLayerNeuronIndex++) {
                for (int prevLayerNeuronIndex = 0; prevLayerNeuronIndex < 
                    this->layerSizes_[layerIndex - 1]; prevLayerNeuronIndex++) {
                    currentLayer->setWeight(currLayerNeuronIndex, prevLayerNeuronIndex,
                        rand(re));
                }
            }
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
                new Layer(layerSizes[i], i == 0 ? layerSizes[0] : layerSizes[i - 1]);
        }
        this->initialiseParams();
    }

    void NeuralNetwork::getOutput(Matrix* input, float* buf, size_t* bufSize)
    {
        // Set values for first layer (input layer)
        float compressedInput[this->layerSizes_[0]];
        for (int i = 0 ;i < this->layerSizes_[0]; i++) { 
            compressedInput[i] = 
                relu(input->get(i / 28, i % 28));
        }

        this->layers_[0]->setNeuronValues(compressedInput);

        for (int layerIndex = 1; layerIndex < this->numLayers_; layerIndex++) {
            float* prevLayerNeuronValues;
            this->layers_[layerIndex - 1]->getNeuronValues(&prevLayerNeuronValues);
            this->layers_[layerIndex]->computeNeuronValues(&prevLayerNeuronValues);
            delete[] prevLayerNeuronValues;
        }

        // Get output layer vector
        this->layers_[this->numLayers_ - 1]->getNeuronValues(&buf);
        (*bufSize) = this->layerSizes_[this->numLayers_ - 1];
    }

    Layer* NeuralNetwork::getLayer(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= this->numLayers_)
            throw std::invalid_argument("Layer index is invalid");

        return this->layers_[layerIndex];
    }

    void NeuralNetwork::train()
    {
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
    void printSeperator(int numCols, int *colSpaces) 
    {
        for (int colIndex = 0; colIndex < numCols; colIndex++) {
            std::cout << "_";
            for (int i = 0; i < colSpaces[colIndex]; i++) { std:: cout << "_"; }
        }
        std::cout << "_\n";
    }

    float relu(double raw)
    {
        return (raw > 0) ? raw : 0;
    }

    double dotProduct(double* vectorA, float* vectorB, int length)
    {
        double sum = 0;
        for (int i = 0; i < length; i++) {
            sum += vectorA[i] * vectorB[i];
        }
        return sum;
    }
}
