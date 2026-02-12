#pragma once

#include <vector>

namespace mlfs {

class Matrix {
private:
    int numRows_;
    int numCols_;
    std::vector<std::vector<double>> data_;

public:
    Matrix(int numRows, int numCols);

    Matrix(std::vector<std::vector<double>>& data);

    double get(int r, int c);

    void set(int r, int c, double val);
};

}
