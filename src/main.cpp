#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector>
#include <cerrno>
#include <cstring>
#include <chrono>

#include "Matrix.h"
#include "models/LinearRegression.h"

enum ModelType {
    NONE,
    LINEAR_REGRESSION
};

struct CliParams {
    ModelType model;
    CliParams() { model = ModelType::NONE; }
};

struct Data {
    Matrix testX;
    Matrix testY;
    Matrix trainX;
    Matrix trainY;
};

// ===============================================================================================
// FUNCTION DECLARATIONS
// ===============================================================================================
CliParams parseCliArgs(int argc, char* argv[]);

// Helper Functions
Data getBostonData();

// ===============================================================================================
// MAIN FUNCTION
// ===============================================================================================
int main(int argc, char* argv[])
{
    CliParams cliParams = parseCliArgs(argc, argv);

    switch (cliParams.model) {
        case ModelType::LINEAR_REGRESSION:
            Data data = getBostonData();
            LinearRegression model(1);
            auto start = std::chrono::high_resolution_clock::now();
            model.train(data.trainX, data.trainY);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            std::cout << "Implementation Training Time: " << std::fixed << std::setprecision(4)
            << duration.count() << " Milliseconds\n";
            model.test(data.testX, data.testY);
            break;
    }

    return 0;
}

// ===============================================================================================
// FUNCTION IMPLEMENTATIONS
// ===============================================================================================
CliParams parseCliArgs(int argc, char* argv[])
{
    argc--;
    argv++;

    if (argc == 0)
        throw std::invalid_argument("You must provide a model");

    CliParams cliParams;
    while (argc > 0) {
        std::string currentArg(argv[0]);

        if (currentArg == "-linReg") {
            cliParams.model = ModelType::LINEAR_REGRESSION;
        } else if (cliParams.model != ModelType::NONE) {
            throw std::invalid_argument("You cannot specify more than one model");
        } else {
            throw std::invalid_argument("Unknown argument: " + currentArg);
        }

        argc--;
        argv++;
    }

    if (cliParams.model == ModelType::NONE)
        throw std::invalid_argument("You must specify a model");

    return cliParams;
}

Data getBostonData()
{
    std::ifstream file("../data/Boston.csv");

    std::vector<std::vector<double>> trainXData;
    std::vector<std::vector<double>> trainYData;

    bool skippedHeader = false;
    std::string line;
    int numOfObs = 0;
    while (std::getline(file, line)) {
        if (!skippedHeader) {
            skippedHeader = true;
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        int cellIndex = 0;
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) {
            if (cellIndex == 1) {
                row.push_back(std::stod(cell));
            }

            if (cellIndex == 14) {
                trainYData.push_back({ std::stod(cell) });
            }

            cellIndex++;
        }

        trainXData.push_back(row);
        numOfObs++;
    }

    Data data;
    data.trainX = Matrix(trainXData.size(), 1, trainXData);
    data.trainY = Matrix(trainYData.size(), 1, trainYData);
    data.testX = data.trainX;
    data.testY = data.trainY;

    file.close();

    return data;
}
