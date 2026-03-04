#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/linear_regression.hpp>
#include <jmll/models/tuning/evaluation_metric.hpp>
#include <jmll/models/tuning/fitting_policy.hpp>
#include <jmll/models/tuning/grid_searcher.hpp>

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>

// ===============================================================================================
// CONSTANTS AND TYPES
// ===============================================================================================
const double TRAIN_SPLIT = 0.8;

const std::vector<int> LIN_REG_IGNORE_INDEXES = { 0, 3, 7 };

const std::string BOSTON_FILEPATH = "../data/Boston.csv";
const int BOSTON_N = 506;

const std::string USAGE_MSG = 
    "[Usage]: ./jmll [model_type]\n"
    "Model Types:\n"
    "Linear Regression: -linReg";

enum ModelType {
    NONE,
    LINEAR_REGRESSION
};

struct Data {
    jmll::core::Matrix xTrain;
    jmll::core::Matrix xTest;
    jmll::core::Vector yTrain;
    jmll::core::Vector yTest;
};

// ===============================================================================================
// FUNCTION DECLARATIONS
// ===============================================================================================
ModelType parseCliArguments(int argc, char* argv[]);
void throwUsageError();
Data getData(ModelType modelType);
void runModel(ModelType modelType, Data data);

Data getBostonData(std::vector<int> ignoreIndexes);


// ===============================================================================================
// MAIN FUNCTION
// ===============================================================================================
int main(int argc, char* argv[]) {
    ModelType modelType = parseCliArguments(argc, argv);
    Data data = getData(modelType);
    runModel(modelType, data);

    return 0;
}

// ===============================================================================================
// HELPERS
// ===============================================================================================
ModelType parseCliArguments(int argc, char* argv[]) {
    // Skip program name 
    argc--;
    argv++;

    // Retrieve model type
    ModelType modelType = ModelType::NONE;
    while (argc > 0) {
        std::string current(argv[0]);

        if (modelType != ModelType::NONE) throwUsageError();
        else if (current == "-linReg") modelType = ModelType::LINEAR_REGRESSION;
        else throwUsageError();

        argc--;
        argv++;
    }

    if (modelType == ModelType::NONE) throwUsageError();

    return modelType;
}

void throwUsageError() {
    throw new std::invalid_argument(USAGE_MSG);
}

Data getData(ModelType modelType) {
    switch (modelType) {
        case ModelType::LINEAR_REGRESSION:
            return getBostonData(LIN_REG_IGNORE_INDEXES);
    }
}

Data getBostonData(std::vector<int> ignoreIndexes) {
    std::unordered_set<int> ignore(ignoreIndexes.begin(), ignoreIndexes.end());

    std::vector<std::vector<double>> xTrainData;
    std::vector<std::vector<double>> xTestData;
    std::vector<double> yTrainData;
    std::vector<double> yTestData;

    std::ifstream file(BOSTON_FILEPATH);
    if (!file.is_open()) throw new std::runtime_error("Data File could not be found");

    std::string line;
    int lineIndex = -1;
    while (std::getline(file, line)) {
        if (lineIndex == -1) {
            lineIndex++;
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        int cellIndex = 0;
        std::vector<double> row;

        while(std::getline(ss, cell, ',')) {
            if (!ignore.contains(cellIndex)) {
                row.push_back(std::stod(cell));
            }

            cellIndex++;
        }

        if (lineIndex < BOSTON_N * TRAIN_SPLIT) {
            yTrainData.push_back(row.back());
            row.pop_back();
            xTrainData.push_back(row);
        } else {
            yTestData.push_back(row.back());
            row.pop_back();
            xTestData.push_back(row);
        }

        lineIndex++;
    }

    return {
        jmll::core::Matrix(xTrainData),
        jmll::core::Matrix(xTestData),
        jmll::core::Vector(yTrainData),
        jmll::core::Vector(yTestData)
    };
}

void runModel(ModelType modelType, Data data) {

    auto start = std::chrono::high_resolution_clock::now();

    switch(modelType) {
        case ModelType::LINEAR_REGRESSION:
            jmll::models::tuning::GridSearcher<
                jmll::models::LinearRegression<jmll::models::tuning::Ridge>
            > gridSearcher;

            std::vector<double> lambdas = { 0.01, 0.1, 1, 2, 5, 10 };
            jmll::models::LinearRegression<jmll::models::tuning::Ridge> model 
                = gridSearcher.get(data.xTrain, data.yTest, lambdas);
            jmll::core::Vector yPred = model.predict(data.xTest);

            double rSquared = model.evaluate(yPred, data.yTest);

            std::cout << "Implementation R Squared Value: " << std::fixed << std::setprecision(2)
                << rSquared << std::endl;

            break;
    }
}
