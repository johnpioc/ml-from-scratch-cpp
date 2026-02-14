#pragma once

#include <vector>

namespace mlfs {
namespace core {

class Vector {
private:
    /* The underlying data inside the vector */
    std::vector<double> data_;

public:
    /* Flag to represent if this is a column or row vector */
    bool isColVector = true;

    /* Number of cells in this vector */
    int numCells;

    /* Initialises a vector with a given number of cells to 0.0 */
    Vector(int numCells);

    /* Uses a given vector of doubles to initialise a Vector */
    Vector(std::vector<double> data);

    /* Sets a given value at a given position in the vector */
    void set(int i, double val);

    /* Retrives a value at a given position in the vector */
    double get(int i);

    /* Col Vector <=> Row Vector */
    void transpose();
};
}
}
