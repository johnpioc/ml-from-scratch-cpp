#include <fstream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <chrono>

#include "Structures.h"

#define OBS_SIZE 28 * 28
#define TOTAL_NUM_OF_OBSERVATIONS 60000
#define OUTPUT_LAYER_SIZE 10

const int DEFAULT_LAYERS[] = { 16, 16 };
const int NUM_DEFAULT_LAYERS = 2;

void processCmdLineArgs(int argc, char* argv[], Structures::NeuralNetworkArgs* buf) {
    argc--;
    argv++;
    buf->layerSizes.push_back(OBS_SIZE);

    if (argc == 0) {
        for (int i = 0; i < NUM_DEFAULT_LAYERS; i++) {
            buf->layerSizes.push_back(DEFAULT_LAYERS[i]);
        }
        buf->layerSizes.push_back(OUTPUT_LAYER_SIZE);
        return;
    }

    while (argc > 0) {
        std::string current(argv[0]);

        if (current == "-l") {
            argc--;
            argv++;

            if (argc == 0)
                throw std::invalid_argument("Please provide layer sizes");

            std::string layerSizesString(argv[0]);
            std::string currentNumberString = "";
            for (int i = 0; i < layerSizesString.size(); i++) {
                char currentChar = layerSizesString[i];
                if (currentChar == ',' || i == layerSizesString.size() - 1) {
                    if (i == layerSizesString.size() - 1)
                        currentNumberString += currentChar;

                    size_t pos;
                    int currentNumber = std::stoi(currentNumberString, &pos);

                    if (currentNumber <= 0) {
                        throw std::invalid_argument("layer size must be greater than 0");
                    }

                    if (pos != currentNumberString.length()) {
                        throw std::invalid_argument("layer size invalid: " + currentNumberString);
                    }

                    buf->layerSizes.push_back(currentNumber);
                    currentNumberString = "";
                } else {
                    currentNumberString += currentChar;
                }
            }
            buf->layerSizes.push_back(OUTPUT_LAYER_SIZE);
        } else {
            throw std::invalid_argument("Invalid command line option: " + current);
        }

        argc--;
        argv++;
    }
}

void printParams(Structures::NeuralNetworkArgs& args)
{
    std::cout << "Neural Network Arguments: \n\n";
    std::cout << "Number of Layers: " << args.layerSizes.size() << "\n";
    for (int i = 1; i < args.layerSizes.size() - 1; i++) {
        std::cout << "Hidden Layer " << i << " size: " << args.layerSizes[i] << "\n";
    }
    std::cout << "\nTraining Observations: " << args.numOfTrainingObservations << "\n";
    std::cout << "Test Observations: " << args.numOfTestObservations << "\n";
}

void processTrainingData(std::vector<Structures::Matrix>& observations,
    std::vector<Structures::Matrix>& expected, int n)
{
    std::ifstream trainingFile("./data/mnist_train.csv");
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
    std::ifstream testFile("./data/mnist_test.csv");
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

int main(int argc, char* argv[])
{
    Structures::NeuralNetworkArgs nnetArgs;
    nnetArgs.numOfTrainingObservations = 60000;
    nnetArgs.numOfTestObservations = 10000;
    processCmdLineArgs(argc, argv, &nnetArgs);
    printParams(nnetArgs);

    std::vector<Structures::Matrix> observations;
    std::vector<Structures::Matrix> expected;
    processTrainingData(observations, expected, nnetArgs.numOfTrainingObservations);

    Structures::NeuralNetwork nnet(nnetArgs);
    auto trainingStartTime = std::chrono::high_resolution_clock::now();
    nnet.train(observations, expected);
    auto trainingEndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> trainingRuntime = trainingEndTime - trainingStartTime;
    std::cout << "Training completed in " << std::fixed << std::setprecision(2) <<
        trainingRuntime.count() << " seconds.\n";

    std::vector<Structures::Matrix> testObservations;
    std::vector<double> testExpected;
    processTestData(testObservations, testExpected, nnetArgs.numOfTestObservations);

    nnet.test(testObservations, testExpected);
    return 0;
}
