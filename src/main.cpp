#include <fstream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <chrono>

#include "Structures.h"

#define OBS_SIZE 28 * 28
#define TOTAL_NUM_OF_OBSERVATIONS 60000
#define OUTPUT_LAYER_SIZE 10

void processTrainingData(std::vector<Structures::Matrix>& observations,
    std::vector<Structures::Matrix>& expected, int n)
{
    std::ifstream trainingFile("../data/mnist_train.csv");
    if (!trainingFile.is_open()) {
        throw std::runtime_error("training csv file cannot be opened");
    }

    for (int batchIndex = 0; batchIndex < (n / BATCH_SIZE); batchIndex++) {
        std::vector<std::vector<double>> observationValues(BATCH_SIZE,
            std::vector<double>(OBS_SIZE));
        std::vector<std::vector<double>> expectedValues(BATCH_SIZE,
            std::vector<double>(OUTPUT_LAYER_SIZE, 0.0));

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
                    observationValues[obsIndex][i - 1] = std::stod(value) / 255.0;
                }
            }
        }

        observations.push_back(Structures::Matrix(BATCH_SIZE, OBS_SIZE, observationValues)
            .transpose());
        expected.push_back(Structures::Matrix(BATCH_SIZE, OUTPUT_LAYER_SIZE, expectedValues)
            .transpose());
    }

    trainingFile.close();
}

void processTestData(std::vector<Structures::Matrix>& observations,
    std::vector<double>& expected, int n)
{
    std::ifstream testFile("../data/mnist_test.csv");
    if (!testFile.is_open()) {
        throw std::runtime_error("Test csv file cannot be opened");
    }

    for (int batchIndex = 0; batchIndex < (n / BATCH_SIZE); batchIndex++) {
        std::vector<std::vector<double>> observationValues(BATCH_SIZE,
            std::vector<double>(OBS_SIZE));

        std::string line;
        for (int obsIndex = 0; obsIndex < BATCH_SIZE; obsIndex++) {
            std::getline(testFile, line);
            std::stringstream lineStream(line);
            std::string value;

            for (int i = 0; i < OBS_SIZE + 1; i++) {
                std::getline(lineStream, value, ',');
                double number = std::stod(value);
                if (i == 0) {
                    expected.push_back(number);
                } else {
                    observationValues[obsIndex][i - 1] = number / 255;
                }
            }
        }

        observations.push_back(Structures::Matrix(BATCH_SIZE, OBS_SIZE, observationValues)
            .transpose());
    }

    testFile.close();
}

int main()
{
    size_t numLayers = 4;
    size_t layerSizes[] = { 784, 16, 16, 10 };

    std::vector<Structures::Matrix> observations;
    std::vector<Structures::Matrix> expected;
    processTrainingData(observations, expected, 60000);

    Structures::NeuralNetwork nnet(numLayers, layerSizes);
    auto trainingStartTime = std::chrono::high_resolution_clock::now();
    nnet.train(observations, expected);
    auto trainingEndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> trainingRuntime = trainingEndTime - trainingStartTime;
    std::cout << "Training took " << (trainingRuntime.count()) << " seconds to complete for ";
    std::cout << "60,000 observations\n";

    std::vector<Structures::Matrix> testObservations;
    std::vector<double> testExpected;
    processTestData(testObservations, testExpected, 10000);

    nnet.test(testObservations, testExpected);
    return 0;
}
