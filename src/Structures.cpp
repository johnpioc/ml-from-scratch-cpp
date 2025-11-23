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
    // HELPER FUNCTION DECLARATIONS
    void printSeperator(int numCols, int* colSpaces);
    double dotProduct(double* vectorA, float* vectorB, int length);
    double relu(double raw);
    double reluDerivative(double raw);

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

    void Matrix::getRow(int rowIndex, double** buf)
    {
        (*buf) = new double[this->numCols_];

        for (int colIndex = 0; colIndex < this->numCols_; colIndex++) {
            (*buf)[colIndex] = this->get(rowIndex, colIndex);
        }
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
    void Layer::applySoftmaxDistribution(double* raw)
    {
        double denom = 0;
        for (int i = 0; i < this->neuronsSize_; i++) {
            denom += std::exp(raw[i]);
        }

        for (int i = 0; i < this->neuronsSize_; i++) {
            this->neuronValues_[i] = std::exp(raw[i]) / denom;
        }
    }
    Layer::Layer(size_t neuronsSize, size_t prevLayerNeuronsSize)
    {
        this->neuronsSize_ = neuronsSize;
        this->prevLayerNeuronsSize_ = prevLayerNeuronsSize;

        this->neuronValues_ = new float[this->neuronsSize_];
        this->weights_ = new Matrix(this->neuronsSize_, this->prevLayerNeuronsSize_);
        this->bias_ = new double[this->neuronsSize_];
    }

    size_t Layer::getNeuronsSize() { return this->neuronsSize_; }
    size_t Layer::getPrevLayerNeuronsSize() { return this->prevLayerNeuronsSize_; }

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
        double* raw = new double[this->getNeuronsSize()];
        for (int i = 0; i < this->neuronsSize_; i++) {
            double* currentWeights;
            this->weights_->getRow(i, &currentWeights);
            raw[i] = 
                relu(
                    dotProduct(
                        currentWeights, (*prevLayerNeuronValues),
                        this->prevLayerNeuronsSize_
                    ) - this->bias_[i]
                );
            delete[] currentWeights;
        }

        this->applySoftmaxDistribution(raw);
        delete[] raw;
    }

    void Layer::setNeuronValues(double* values)
    {
        this->applySoftmaxDistribution(values);
    }

    float Layer::getNeuronValue(int neuronIndex)
    {
        if (neuronIndex < 0 || neuronIndex >= this->neuronsSize_)
            throw std::invalid_argument("invalid neuron index");

        return this->neuronValues_[neuronIndex];
    }

    void Layer::getNeuronValues(float** buf)
    {
        for (int i = 0; i < this->neuronsSize_; i++) {
            (*buf)[i] = this->getNeuronValue(i);
        }
    }

    void Layer::setBias(int neuronIndex, double bias) { this->bias_[neuronIndex] = bias; }
    double Layer::getBias(int neuronIndex) { return this->bias_[neuronIndex]; }

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
        for (int i = 0; i < this->layerSizes_[0]; i++) {
            for (int j = 0; j < this->layerSizes_[0]; j++) {
                inputLayer->setWeight(i, j, 1.0);
            }
            inputLayer->setBias(i, 0.0);
        }

        double lowerBound = -1000.0;
        double upperBound = 1000.0;
        std::uniform_real_distribution<double> rand(lowerBound, upperBound);
        std::default_random_engine re;
        re.seed(67);

        // For rest of layers, apply a random weight and bias
        for (int layerIndex = 1; layerIndex < this->numLayers_; layerIndex++) {
            Layer* currentLayer = this->layers_[layerIndex];
            for (int currLayerNeuronIndex = 0; currLayerNeuronIndex < 
                this->layerSizes_[layerIndex]; currLayerNeuronIndex++) {
                for (int prevLayerNeuronIndex = 0; prevLayerNeuronIndex < 
                    this->layerSizes_[layerIndex - 1]; prevLayerNeuronIndex++) {
                    currentLayer->setWeight(currLayerNeuronIndex, prevLayerNeuronIndex,
                        rand(re));
                }
                currentLayer->setBias(currLayerNeuronIndex, rand(re));
            }
        }
    }

    double NeuralNetwork::cost(int layerIndex, int neuronIndex, double expected)
    {
        return pow(expected - this->layers_[layerIndex]->getNeuronValue(neuronIndex), 2.0);
    }

    void NeuralNetwork::backPropagate(double* expectedOutput, Matrix** weightsBuf, double***
        biasBuf)
    {
        double* expected = new double[this->layerSizes_[this->numLayers_ - 1]];
        for (int i = 0; i < this->layerSizes_[this->numLayers_ - 1]; i++) {
            expected[i] = expectedOutput[i];
        }

        for (int layerIndex = this->numLayers_ - 1; layerIndex > 0; layerIndex--) {
            Layer* currentLayer = this->getLayer(layerIndex);
            double* nextExpected = new double[currentLayer->getNeuronsSize()];
            for (int currLayerIndex = 0; currLayerIndex < currentLayer->getNeuronsSize(); 
                currLayerIndex++) {
                double dc_da = 2 * (currentLayer->getNeuronValue(currLayerIndex) - 
                    expected[currLayerIndex]);
                double da_dz = 
                    reluDerivative(currentLayer->getNeuronValue(currLayerIndex));
                nextExpected[currLayerIndex] = dc_da * da_dz;
                for (int prevLayerIndex = 0; prevLayerIndex < 
                    currentLayer->getPrevLayerNeuronsSize(); prevLayerIndex++) {
                    double dz_dw = 
                        this->layers_[layerIndex - 1]->getNeuronValue(prevLayerIndex);
                    weightsBuf[layerIndex]->put(currLayerIndex, prevLayerIndex,
                        dc_da * da_dz * dz_dw);
                }

                (*biasBuf)[layerIndex][currLayerIndex] = dc_da * da_dz;
            }

            delete[] expected;
            if (layerIndex > 1) {
                Layer* nextLayer = this->layers_[layerIndex - 1];
                expected = new double[nextLayer->getNeuronsSize()]();
                for (int i = 0; i < currentLayer->getNeuronsSize(); i++) {
                    for (int j = 0; j < nextLayer->getNeuronsSize(); j++) {
                        expected[j] += nextExpected[i] * currentLayer->getWeight(i,j);
                    }
                }
            }
            delete[] nextExpected;
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

    void NeuralNetwork::getOutput(float** buf)
    {
        Layer* outputLayer = this->layers_[this->numLayers_ - 1];
        for (int i = 0; i < outputLayer->getNeuronsSize(); i++) {
            (*buf)[i] = outputLayer->getNeuronValue(i);
        }
    }

    void NeuralNetwork::feedForward(Matrix* input)
    {
        // Manually set the values in the first layer
        double* inputVector = new double[this->layerSizes_[0]];
        for (int i = 0; i < this->layerSizes_[0]; i++) {
            inputVector[i] = input->get(i / 28, i % 28);
        }
        this->getLayer(0)->setNeuronValues(inputVector);
        delete[] inputVector;

        // Traverse from the second layer to end and compute neuron values 
        for (int i = 1; i < this->numLayers_; i++) {
            Layer* currentLayer = this->layers_[i];
            float* prevLayerNeuronValues = new float[this->layerSizes_[i - 1]];
            this->layers_[i - 1]->getNeuronValues(&prevLayerNeuronValues);
            currentLayer->computeNeuronValues(&prevLayerNeuronValues);
            delete[] prevLayerNeuronValues;
        }
    }

    Layer* NeuralNetwork::getLayer(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= this->numLayers_)
            throw std::invalid_argument("Layer index is invalid");

        return this->layers_[layerIndex];
    }

    void NeuralNetwork::train(std::vector<Image*> observations, int n)
    {
        Matrix** finalWeights = new Matrix*[this->numLayers_];
        double** finalBiases = new double*[this->numLayers_];
        for (int i = 1; i < this->numLayers_; i++) {
            size_t numRows = this->layers_[i]->getNeuronsSize();
            size_t numCols = this->layers_[i]->getPrevLayerNeuronsSize();
            finalWeights[i] = new Matrix(numRows, numCols);
            finalBiases[i] = new double[numRows];
        }

        for (int i = 0; i < n; i++) {
            Image* current = observations[i];
            int expected = current->getNumber();
            double* expectedOutput = new double[this->layerSizes_[this->numLayers_ - 1]]();
            expectedOutput[expected - 1] = 1.0;

            Matrix** weightsBuf = new Matrix*[this->numLayers_];
            double** biasBuf = new double*[this->numLayers_];

            for (int layerIndex = 1; layerIndex < this->numLayers_; layerIndex++) {
                size_t numRows = this->layers_[layerIndex]->getNeuronsSize();
                size_t numCols = this->layers_[layerIndex]->getPrevLayerNeuronsSize();
                weightsBuf[layerIndex] = new Matrix(numRows, numCols);

                biasBuf[layerIndex] = new double[numRows]();
            }

            this->feedForward(current->getCellValues());
            this->backPropagate(expectedOutput, weightsBuf, &biasBuf);

            for (int layerIndex = 1; layerIndex < this->numLayers_; layerIndex++) {
                size_t numRows = this->layers_[layerIndex]->getNeuronsSize();
                size_t numCols = this->layers_[layerIndex]->getPrevLayerNeuronsSize();

                for (int r = 0; r < numRows; r++) {
                    for (int c = 0; c < numCols; c++) {
                        double weightDelta = weightsBuf[layerIndex]->get(r, c);
                        double currentWeight = finalWeights[layerIndex]->get(r,c);
                        finalWeights[layerIndex]->put(r, c, currentWeight + weightDelta);
                    }

                    double biasDelta = biasBuf[layerIndex][r];
                    finalBiases[layerIndex][r] += biasDelta;
                }
            }
        }

        for (int i = 1; i < this->numLayers_; i++) {
            size_t numRows = this->layers_[i]->getNeuronsSize();
            size_t numCols = this->layers_[i]->getPrevLayerNeuronsSize();
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    this->layers_[i]->setWeight(r, c, finalWeights[i]->get(r, c));
                }
                this->layers_[i]->setBias(r, finalBiases[i][r]);
            }
            delete finalWeights[i];
            delete[] finalBiases[i];
        }

        delete[] finalWeights;
        delete[] finalBiases;
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

    /**
     * @brief Performs the dot product on two vectors
     * @param vectorA an array of doubles
     * @param vectorB an array of doubles
     * @param length the size of vectorA and vectorB
     * @returns the dot product of vectorA and vectorB
     */
    double dotProduct(double* vectorA, float* vectorB, int length)
    {
        double sum = 0;
        for (int i = 0; i < length; i++) {
            sum += vectorA[i] * vectorB[i];
        }
        return sum;
    }

    /** Rectified linear unit (squishification function for neuron values) */
    double relu(double raw)
    {
        return (raw > 0) ? raw : 0;
    }

    /** Derivative of the relu */
    double reluDerivative(double raw)
    {
        return (raw > 0) ? 1 : 0;
    }
}
