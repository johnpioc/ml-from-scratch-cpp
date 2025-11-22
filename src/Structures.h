#pragma once

#include <iostream>
#include <cstdint>

namespace Structures {
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
        double** data_;

    public:
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
        Matrix(size_t numRows, size_t numCols, double** values);

        /** Returns the number of rows inside the matrix */
        size_t getNumRows();
        /** Returns the number of columns inside the matrix */
        size_t getNumCols();
        /** Returns all the cell values inside the matrix as an array of double arrays */
        double** getData();

        /**
         * @brief Returns a cell value 
         * @param rowIndex the index of the row of the cell we want to get the value from 
         * @param colIndex the index of the column of the cell we want to get the value from
         * @return the cell value corresponding to the row and column index
         */
        double get(int rowIndex, int colIndex);

        /**
         * @brief returns the given values of a row in the matrix
         * @param rowIndex the index of the row we want to get the values from
         * @param buf the pointer to an allocated array of doubles that will be storing the row
         * values
         */
        void getRow(int rowIndex, double** buf);

        /**
         * @brief inserts/updates a given cell in the matrix
         * @param rowIndex the index of the row of the cell we want to update
         * @param colIndex the index of the column of the cell we want to update
         * @param value the value we want to store in the cell
         */
        void put(int rowIndex, int colIndex, double value);

        /** Prints the contents of the matrix to stdout */
        void print();

        /** Destructor */
        ~Matrix();
    };

    /**
     * Layer of neurons inside a Neural Network
     */
    struct Layer {
    private:
        // Number of neurons inside the layer
        size_t neuronsSize_;
        // Number of neurons in previous layer
        size_t prevLayerNeuronsSize_;
        // Matrix of connection weights from the previous layer to this layer
        Matrix* weights_;
        // The values of the neurons of this layer
        double* neuronValues_;
        // The bias associated with this layer
        double bias_;

        /**
         * @brief Generates a probability distribution of all the neurons inside this layer with
         * their corresponding values
         * @param buf a pointer to an allocated array of floats which will store the probability 
         * distribution values for each neuron
         */
        void getSoftmaxDistribution(float** buf);

    public:
        /**
         * @brief constructs a Layer struct and initialises neuron values to 0.0
         * @param neuronsSize the number of neurons inside the layer
         * @param prevLayerNeuronsSize the number of neurons inside the previous layer
         */
        Layer(size_t neuronsSize, size_t prevLayerNeuronsSize);

        /**
         * @brief Sets the weight for a given connection between a neuron in the previous layer and
         * a neuron in this layer
         * @param currLayerNeuronIndex the index of the neuron in this layer
         * @param prevLayerNeuronIndex the index of the neuron in the previous layer
         * @param value the weight of the connection between the two neurons
         */
        void setWeight(int currLayerNeuronIndex, int prevLayerNeuronIndex, double value);

        /**
         * @brief Retrives the weight for a given connection between a neuron in the previous layer
         * and a neuron in this layer
         * @param currLayerNeuronIndex the index of the neuron in this layer
         * @param prevLayerNeuronIndex the index of the neuron in the previous layer
         * @returns a double value representing the weight of the connection
         */
        double getWeight(int currLayerNeuronIndex, int prevLayerNeuronIndex);

        /**
         * @brief Computes and saves the neuron values for this layer based on connection weights
         * and a given array of neuron values from the previous layer
         * @param prevLayerNeuronValues a pointer to an array of doubles that represent the neuron
         * values from the previous layer
         */
        void computeNeuronValues(double** prevLayerNeuronValues);

        /**
         * @brief Directly sets the neuron values inside this layer
         * @param values a pointer to an array of doubles that represent the values for the neurons
         * in this layer
         */
        void setNeuronValues(double** values);

        /**
         * @brief Retrives the value of a given neuron
         * @param neuronIndex the index of the neuron in this layer who's value we want to retrieve
         */
        float getNeuronValue(int neuronIndex);

        /**
         * @brief Retrives the values of all the neurons in the layer
         * @param buf a pointer to an allocated array of doubles which will store the values of the
         * neurons in this layer
         */
        void getNeuronValues(double** buf);

        /** Sets the bias associated with the layer */
        void setBias(double bias);
        /** Returns the bias associated with the layer */
        double getBias();

        /** Destructor */
        ~Layer();
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

        /**
         * @brief computes the cost of a given neuron inside the network
         * @param layerIndex the index of the layer that contains the neuron of interest
         * @param neuronIndex the index of the neuron inside the layer
         * @param expected the expected value of the neuron of interest
         * @returns a double value which represents the cost between the neuron's actual value
         * versus the expected
         */
        double cost(int layerIndex, int neuronIndex, double expected);

        /**
         * @brief Performs the back-propagation algorithm by taking in an array of expected output
         * neuron values and traverses each layer from output to input adjusting weights and biases
         */
        void backPropagate(double* expectedOutput);

    public:
        /**
         * @brief Constructs a Neural Network struct along with its associated layers
         * @param numLayers the number of layers inside this neural network
         * @param layerSizes an array of size_t values that represent the number of neurons for each
         * layer in the network
         */
        NeuralNetwork(size_t numLayers, size_t* layerSizes);

        /**
         * @brief Retrives the output layer neuron values formatted as a probability distribution
         * @param buf a pointer to an allocated array of floats which will store the softmaxed
         * values for each neuron
         */
        void getOutput(float** buf);

        /**
         * @brief Performs the feed-forward algorithm, taking in a matrix of inputs for the first
         * layer, then computes and saves neuron values for subsequent layers up to the output
         */
        void feedForward(Matrix* input);

        /** Retrives a layer from the network */
        Layer* getLayer(int layerIndex);

        /** Destructor */
        ~NeuralNetwork();
    };
}
