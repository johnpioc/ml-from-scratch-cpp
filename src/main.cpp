#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector>
#include <cerrno>
#include <cstring>
#include <chrono>

#include "Matrix.h"
#include "models/Model.h"
#include "models/LinearRegression.h"
#include "models/LogisticRegression.h"
#include "models/QuadraticDiscriminantAnalysis.h"

enum ModelType {
    NONE,
    LINEAR_REGRESSION,
    LOGISTIC_REGRESSION,
    QDA
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
Model* getModel(ModelType modelType);
Data getData(ModelType modeltype);

// Helper Functions
Data getBostonData();
Data getStockMarketData(int numOfLags);

// ===============================================================================================
// MAIN FUNCTION
// ===============================================================================================
int main(int argc, char* argv[])
{
    CliParams cliParams = parseCliArgs(argc, argv);
    Model* model = getModel(cliParams.model);
    Data data = getData(cliParams.model);

    auto start = std::chrono::high_resolution_clock::now();

    model->train(data.trainX, data.trainY);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Implementation Training Time: " << std::fixed << std::setprecision(4)
        << duration.count() << " Milliseconds\n";

    model->test(data.testX, data.testY);
    delete model;
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
        } else if (currentArg == "-logReg") {
            cliParams.model = ModelType::LOGISTIC_REGRESSION;
        } else if (currentArg == "-qda") {
            cliParams.model = ModelType::QDA;
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

Model* getModel(ModelType modelType)
{
    Model* model;

    switch(modelType) {
        case ModelType::LINEAR_REGRESSION:
            model = new LinearRegression();
            break;
        case ModelType::LOGISTIC_REGRESSION:
            model = new LogisticRegression();
            break;
        case ModelType::QDA:
            model = new QuadraticDiscriminantAnalysis(2);
            break;
    }

    return model;
}

Data getData(ModelType modelType)
{
    Data data;

    switch (modelType) {
        case ModelType::LINEAR_REGRESSION:
            data = getBostonData();
            break;
        case ModelType::LOGISTIC_REGRESSION:
            data = getStockMarketData(1);
            break;
        case ModelType::QDA:
            data = getStockMarketData(2);
            break;
    }

    return data;
}

Data getBostonData()
{
    std::ifstream file("../data/Boston.csv");

    std::vector<std::vector<double>> trainXData;
    std::vector<std::vector<double>> trainYData;

    bool skippedHeader = false;
    std::string line;
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
    }

    Data data;
    data.trainX = Matrix(trainXData.size(), 1, trainXData);
    data.trainY = Matrix(trainYData.size(), 1, trainYData);
    data.testX = data.trainX;
    data.testY = data.trainY;

    file.close();

    return data;
}

Data getStockMarketData(int numOfLags)
{
    std::ifstream file("../data/SMarket.csv");

    std::vector<std::vector<double>> trainXData;
    std::vector<std::vector<double>> trainYData;

    bool skippedHeader = false;
    std::string line;
    while (std::getline(file, line)) {
        if (!skippedHeader) {
            skippedHeader = true;
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        int cellIndex = 0;
        while (std::getline(ss, cell, ',')) {
            if (cellIndex > 1 && (cellIndex - 2) < numOfLags) {
                row.push_back(std::stod(cell));
            }

            if (cellIndex == 9) {
                trainYData.push_back({ cell == "\"Up\"" ? 0.0 : 1.0 });
            }

            cellIndex++;
        }

        trainXData.push_back(row);
    }

    Data data;
    data.trainX = Matrix(trainXData.size(), numOfLags, trainXData);
    data.trainY = Matrix(trainYData.size(), numOfLags, trainYData);
    data.testX = data.trainX;
    data.testY = data.trainY;

    file.close();

    return data;
}
