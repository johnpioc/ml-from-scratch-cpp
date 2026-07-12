#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/OLS.hpp>
#include <sstream>
#include <vector>

// ===============================================================================================
// CONSTANTS
// ===============================================================================================
using jmll::core::Matrix;
using jmll::core::Vector;
using jmll::models::OLS;

const int SUCCESS_NUM = 0;
const int USAGE_ERROR_NUM = 1;
const std::string USAGE_ERROR_MSG = "Usage Error\n";
const int FILE_ERROR_NUM = 2;
const std::string FILE_ERROR_MSG = "Cannot open data file\n";

const std::string bostonDatasetFilepath = "./data/Boston.csv";

enum ModelToRun { NONE, LINEAR_REGRESSION };

// ===============================================================================================
// FUNCTION DECLARATIONS
// ===============================================================================================
int parseCliArguments(int argc, char* argv[], ModelToRun& modelToRun);
void parseExitCode(int exitCode);
int getBostonDataset(Matrix& X, Vector& y, std::vector<std::string>& columnNames);

// ===============================================================================================
// MAIN FUNCTION
// ===============================================================================================
int main(int argc, char* argv[]) {
    // Parse CLI arguments
    ModelToRun modelToRun = NONE;
    parseExitCode(parseCliArguments(argc, argv, modelToRun));

    // Get Data
    Matrix X(0, 0);
    Vector y;
    std::vector<std::string> columnNames;
    parseExitCode(getBostonDataset(X, y, columnNames));

    // Fit Linear Regression
    OLS model;
    model.fit(X, y);

    return 0;
}

// ===============================================================================================
// HELPERS
// ===============================================================================================
int parseCliArguments(int argc, char* argv[], ModelToRun& modelToRun) {
    // Skip program name
    argc--;
    argv++;

    // Initialise model to none
    modelToRun = NONE;

    // Iterate through every command line argument
    while (argc > 0) {
        std::string current(argv[0]);

        if (modelToRun != NONE) {
            return USAGE_ERROR_NUM;
        } else if (current == "1") {  // linear regression
            modelToRun = LINEAR_REGRESSION;
        } else {
            return USAGE_ERROR_NUM;
        }

        argv++;
        argc--;
    }

    if (modelToRun == NONE) return USAGE_ERROR_NUM;

    return SUCCESS_NUM;
}

void parseExitCode(int exitCode) {
    switch (exitCode) {
        case USAGE_ERROR_NUM: {
            std::cerr << USAGE_ERROR_MSG;
            break;
        }
        case FILE_ERROR_NUM: {
            std::cerr << FILE_ERROR_MSG;
            break;
        }
    }

    if (exitCode != 0) std::exit(exitCode);
}

int getBostonDataset(Matrix& X, Vector& y, std::vector<std::string>& columnNames) {
    const std::vector<int> FEATURE_INDICES = {1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 14};

    std::ifstream file(bostonDatasetFilepath);

    // Check that file opened successfully
    if (!file.is_open()) return FILE_ERROR_NUM;

    std::string line;
    int lineIndex = 0;

    std::vector<std::vector<double>> dataVector;
    std::vector<double> labelVector;

    // Go through each row
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        int fieldIndex = 0;
        std::vector<double> row;

        // Split line by commas
        while (std::getline(ss, field, ',')) {
            if (std::ranges::contains(FEATURE_INDICES, fieldIndex)) {
                if (lineIndex == 0)
                    columnNames.push_back(field);
                else if (fieldIndex == 14)
                    labelVector.push_back(std::stod(field));
                else
                    row.push_back(std::stod(field));
            }

            fieldIndex++;
        }

        if (lineIndex != 0) dataVector.push_back(row);
        lineIndex++;
    }

    X = Matrix(dataVector);
    y = Vector(labelVector);

    return SUCCESS_NUM;
}
