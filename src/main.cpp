#include <chrono>
#include <iomanip>
#include <iostream>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/linear_regression.hpp>
#include <jmll/models/tuning/evaluation_metric.hpp>
#include <jmll/models/tuning/fitting_policy.hpp>
#include <jmll/models/tuning/grid_searcher.hpp>
#include <stdexcept>
#include <string>

#include "jmll/data_preprocessing/datasets.hpp"
#include "jmll/data_preprocessing/train_test_split.hpp"

// ===============================================================================================
// CONSTANTS AND TYPES
// ===============================================================================================
const double TEST_SPLIT = 0.2;

const std::string USAGE_MSG =
    "[Usage]: ./jmll [model_type]\n"
    "Model Types:\n"
    "Linear Regression: -linReg";

enum ModelType { NONE, LINEAR_REGRESSION };

// ===============================================================================================
// FUNCTION DECLARATIONS
// ===============================================================================================
ModelType parseCliArguments(int argc, char* argv[]);
void throwUsageError();
void runModel(ModelType modelType);

// ===============================================================================================
// MAIN FUNCTION
// ===============================================================================================
int main(int argc, char* argv[]) {
    ModelType modelType = parseCliArguments(argc, argv);
    runModel(modelType);

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

        if (modelType != ModelType::NONE)
            throwUsageError();
        else if (current == "-linReg")
            modelType = ModelType::LINEAR_REGRESSION;
        else
            throwUsageError();

        argc--;
        argv++;
    }

    if (modelType == ModelType::NONE) throwUsageError();

    return modelType;
}

void throwUsageError() { throw new std::invalid_argument(USAGE_MSG); }

void runModel(ModelType modelType) {
    using namespace jmll::data_preprocessing;
    using namespace jmll::models;
    using namespace jmll::core;

    auto start = std::chrono::high_resolution_clock::now();

    switch (modelType) {
        case ModelType::LINEAR_REGRESSION:
            TrainTestSplit data = getBostonData(TEST_SPLIT);
            LinearRegression model;
            model.fit(data.xTrain, data.yTrain);
            Vector yPred = model.predict(data.xTest);
            double rSquared = model.evaluate(yPred, data.yTest);

            std::cout << "Implementation R Squared Value: " << std::fixed << std::setprecision(2)
                      << rSquared << std::endl;

            break;
    }
}
