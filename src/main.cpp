#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <sstream>

#include "Structures.h"

#define OBS_SIZE 28 * 28
#define TOTAL_NUM_OF_OBSERVATIONS 60000
#define OUTPUT_LAYER_SIZE 10

void processTrainingData(std::vector<Structures::Matrix> observations,
    std::vector<Structures::Matrix> expected)
{
    std::ifstream trainingFile("../data/mnist_train.csv");
    if (!trainingFile.is_open()) {
        throw std::runtime_error("training csv file cannot be opened");
    }

    for (int batchIndex = 0; batchIndex < TOTAL_NUM_OF_OBSERVATIONS / BATCH_SIZE; batchIndex++) {
        double** observationValues = new double*[BATCH_SIZE];
        double** expectedValues = new double*[BATCH_SIZE];
        for (int i = 0; i < BATCH_SIZE; i++) {
            observationValues[i] = new double[OBS_SIZE];
            expectedValues[i] = new double[OUTPUT_LAYER_SIZE]();
        }

        std::string line;
        for (int obsIndex = 0; obsIndex < BATCH_SIZE; obsIndex++) {
            std::getline(trainingFile, line);
            std::stringstream lineStream(line);
            std::string value;

            for (int i = 0; i < OBS_SIZE + 1; i++) {
                std::getline(lineStream, value, ',');
                if (i == 0) {
                    int expectedNumber = std::stod(value);
                    expectedValues[obsIndex][expectedNumber] = 1.00;
                } else {
                    observationValues[obsIndex][i - 1] = std::stod(value);
                }
            }
        }

        observations.push_back(Structures::Matrix(BATCH_SIZE, OBS_SIZE, observationValues));
        expected.push_back(Structures::Matrix(BATCH_SIZE, OUTPUT_LAYER_SIZE, expectedValues));

        delete[] observationValues;
        delete[] expectedValues;
    }

    trainingFile.close();
}

int main()
{
    size_t numLayers = 4;
    size_t layerSizes[] = { 784, 16, 16, 10 };

    std::vector<Structures::Matrix> observations;
    std::vector<Structures::Matrix> expected;
    processTrainingData(observations, expected);

    Structures::NeuralNetwork nnet(numLayers, layerSizes);
    nnet.train(observations, expected);
    return 0;
}
