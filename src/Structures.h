#pragma once

#include <iostream>
#include <cstdint>

namespace Structures {
    struct Matrix {
    private:
        size_t numRows_;
        size_t numCols_;
        double** data_;

    public:
        Matrix(size_t numRows, size_t numCols);
        Matrix(size_t numRows, size_t numCols, double** values);
        
        size_t getNumRows();
        size_t getNumCols();
        double** getData();
        double get(int rowIndex, int colIndex);
        void getRow(int rowIndex, double** buf, size_t* bufSize);
        void put(int rowIndex, int colIndex, double value);
        void print();

        ~Matrix();
    };

    struct Layer {
    private:
        size_t neuronsSize_;
        size_t prevLayerNeuronsSize_;
        Matrix* weights_;
        float* neuronValues_;
        double bias_;

    public:
        Layer(size_t neuronsSize, size_t prevLayerNeuronsSize);

        void setWeight(int currLayerNeuronIndex, int prevLayerNeuronIndex, double value);
        double getWeight(int currLayerNeuronIndex, int prevLayerNeuronIndex);
        void computeNeuronValues(float** prevLayerNeuronValues);
        void setNeuronValues(float* values);
        float getNeuronValue(int neuronIndex);
        void getNeuronValues(float** buf);
        void setBias(double bias);
        double getBias();

        ~Layer();
    };

    struct NeuralNetwork {
    private:
        size_t numLayers_; // Including input and output layer
        size_t* layerSizes_;
        Layer** layers_;

        void initialiseParams();
        double cost(int layerIndex, int neuronIndex, double expected);
        double relu(double raw);
        void backPropagate(double* expectedOutput);

    public:
        NeuralNetwork(size_t numLayers, size_t* layerSizes);
        void getOutput(Matrix* input, float* buf, size_t* bufSize);
        Layer* getLayer(int layerIndex);
        void train();
        ~NeuralNetwork();
    };
}
