#pragma once

#include <iostream>
#include <cstdint>
#include <vector>

#define BATCH_SIZE 1000
#define LEARNING_RATE 0.001

namespace Structures {

    struct NeuralNetworkArgs {
        std::vector<int> layerSizes;
        int numOfTrainingObservations;
        int numOfTestObservations;
    };

    /**
     * Matrix that stores double values
     */
    struct Matrix {
    private:
        // The number of rows in the matrix
        size_t numRows_;
        // The number of columns in the matrix
        size_t numCols_;
        // The underlying data inside the matrix
        std::vector<std::vector<double>> data_;

    public:
        Matrix();

        /**
         * @brief constructs a new matrix struct and initialises all cell values to 0.0
         * @param numRows the number of rows in the matrix
         * @param numCols the number of columns in the matrix
         */
        Matrix(size_t numRows, size_t numCols);

        /**
         * @brief constructs a new matrix struct 
         * @param numRows the number of rows in the matrix
         * @param numCols the number of columns in the matrix
         * @param values the cell values inside the matrix
         */
        Matrix(size_t numRows, size_t numCols, std::vector<std::vector<double>> values);

        // Operator Overloads
        Matrix operator*(const Matrix& other);
        Matrix operator*(const double scalar);
        Matrix operator+(const Matrix& other);
        Matrix operator-(const Matrix& other);

        Matrix dot(const Matrix& other);

        /**
         * @brief constructs a transposed version of this matrix
         * @returns a new matrix object that is the tranposed version
         */
        Matrix transpose();

        /**
         * Returns true if this matrix and given matrix can be multiplied, otherwise false
         */
        bool canMultiply(const Matrix& other);

        /**
         * Returns true if this matrix and given matrix can be added (or subtracted), otherwise
         * false
         */
        bool canAdd(const Matrix& other);

        /** Returns the number of rows inside the matrix */
        size_t getNumRows();
        /** Returns the number of columns inside the matrix */
        size_t getNumCols();

        /**
         * @brief Returns a cell value 
         * @param rowIndex the index of the row of the cell we want to get the value from 
         * @param colIndex the index of the column of the cell we want to get the value from
         * @return the cell value corresponding to the row and column index
         */
        double get(int rowIndex, int colIndex);

        /**
         * @brief inserts/updates a given cell in the matrix
         * @param rowIndex the index of the row of the cell we want to update
         * @param colIndex the index of the column of the cell we want to update
         * @param value the value we want to store in the cell
         */
        void put(int rowIndex, int colIndex, double value);

        /** Prints the contents of the matrix to stdout */
        void print();
    };

    /**
     * Layer of neurons inside a Neural Network
     */
    struct Layer {
    private:
        // The number of neurons in this layer
        size_t neuronsSize_;
        // The number of neurons in the previous layer
        size_t prevLayerNeuronsSize_;
        // The neuron values for each observation in a batch
        Matrix neuronValues_;
        // The weights to the connections from previous layer to this layer
        Matrix weights_;
        // The bias associated with each neuron in this layer
        Matrix bias_;

    public:
        /**
         * @brief constructs a new layer
         * @param neuronsSize the number of neurons to be stored in this layer
         * @param prevLayerNeuronsSize the number of neurons that are stored in the previous layer
         */
        Layer(size_t neuronsSize, size_t prevLayerNeuronsSize);

        /** Returns a matrix of the neuron values in this layer */
        Matrix getNeuronValues();
        /** Manually sets the neuron values in this layer */
        void setNeuronValues(Matrix buf);
        /** Returns the number of neurons in this layer */
        size_t getNeuronsSize();
        /** Returns the number of neurons in the previous layer */
        size_t getPrevLayerNeuronsSize();
        /** Returns the weights associated with this layer and previous */
        Matrix getWeights();
        /** Sets the weights in the matrix */
        void setWeights(Matrix values);
        void setBias(Matrix bias);

        /**
         * @brief adjusts the weights in this layer
         * @param delta a matrix with identical shape to the weights that will subtract the current
         * weight values
         */
        void adjustWeights(Matrix delta);

        /**
         * @brief adjusts the biases in this layer
         * @param delta a matrix with identical shape to the biases that will subtract the current
         * bias values
         */
        void adjustBias(Matrix delta);

        /**
         * @brief Computes the neuron values for this layer using weights, previous layer neuron
         * values and biases
         */
        void computeNeuronValues(Matrix previousLayerNeuronValues);
    };

    struct NeuralNetwork {
    private:
        // The number of layers inside the network
        size_t numLayers_;
        // The sizes of each layer in the network
        size_t* layerSizes_;
        // The layers inside this network
        Layer** layers_;

        /** Initialises model parameters by generating random values for weights and biases */
        void initialiseParams();

        void backPropagate(Matrix expected);

        /**
         * @brief Performs the feed-forward algorithm, taking in a matrix of inputs for the first
         * layer, then computes and saves neuron values for subsequent layers up to the output
         */
        void feedForward(Matrix batch);

        /** Retrives a layer from the network */
        Layer* getLayer(int layerIndex);

    public:
        /**
         * @brief Constructs a Neural Network struct along with its associated layers
         * @param numLayers the number of layers inside this neural network
         * @param layerSizes an array of size_t values that represent the number of neurons for each
         * layer in the network
         */
        NeuralNetwork(NeuralNetworkArgs& args);

        void train(std::vector<Matrix> observations, std::vector<Matrix> expected);
        void test(std::vector<Matrix> observations, std::vector<double> expected);

        /** Destructor */
        ~NeuralNetwork();
    };
}
