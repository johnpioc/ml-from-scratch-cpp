#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <sstream>

#include "Structures.h"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CELLS 784

struct Image {
private:
    int number_;
    Structures::Matrix* cellValues_;

public:
    Image(int number, double** values) {
        this->number_ = number;
        this->cellValues_ = new Structures::Matrix(NUM_ROWS, NUM_COLS, values);
    }

    int getNumber() { return this->number_; }

    Structures::Matrix* getCellValues() { return this->cellValues_; };
};

std::vector<Image*> processTrainingData() 
{
    std::vector<Image*> data;
    std::string line;

    std::ifstream trainingFile("../data/mnist_train.csv");
    if (!trainingFile.is_open()) {
        throw std::runtime_error("training csv file cannot be opened");
    }

    while (std::getline(trainingFile, line)) {
        double** values = new double*[NUM_COLS];
        for (int i = 0; i < NUM_COLS; i++) {
            values[i] = new double[NUM_ROWS];
        }

        std::stringstream lineStream(line);
        std::string value;
        int number = 0;
        int index = 0;

        while (std::getline(lineStream, value, ',')) {
            if (index == 0) {
                number = std::stod(value);
            } else {
                values[index % NUM_ROWS][index / NUM_COLS] = std::stod(value);
            }
            index++;
        }

        data.push_back(new Image(number, values));
        for (int i = 0; i < NUM_COLS; i++) { delete[] values[i]; }
        delete[] values;
    }

    return data;
}

int main()
{
    /* Process Data into Matrices:
    *   index 0: training set
    *   index 1: test set
    */
    std::vector<Image*> data = processTrainingData();
    data.front()->getCellValues()->print();
    return 0;
}
