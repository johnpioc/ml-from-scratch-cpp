#include <fstream>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/data_preprocessing/datasets.hpp>
#include <jmll/data_preprocessing/train_test_split.hpp>
#include <sstream>
#include <vector>

using namespace jmll::core;

//===============================================================================================
// CONSTANT VALUES
//===============================================================================================
const std::string BOSTON_FILEPATH = "../data/Boston.csv";
const int BOSTON_LABEL_INDEX = 14;

//===============================================================================================
// DATASET FUNCTIONS
//===============================================================================================
namespace jmll::data_preprocessing {

TrainTestSplit getBostonData(double testSplit) {
    std::vector<std::vector<double>> xData;
    std::vector<double> yData;

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

        while (std::getline(ss, cell, ',')) {
            if (cellIndex++ == 0) continue;

            if (cellIndex == BOSTON_LABEL_INDEX)
                yData.push_back(std::stod(cell));
            else
                row.push_back(std::stod(cell));
        }

        xData.push_back(row);
    }

    Matrix x(xData);
    Vector y(yData);

    return TrainTestSplit(x, y, testSplit);
}

}  // namespace jmll::data_preprocessing
