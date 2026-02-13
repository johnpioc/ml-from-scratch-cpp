#pragma once

#include <vector>

namespace mlfs {
namespace core {

class Vector {
private:
    /* The underlying data inside the vector */
    std::vector<double> data_;

    /* Flag to represent if this is a column or row vector */
    bool isColVector_ = true;

    /* Number of cells in this vector */
    int numCells_;

public:
    /* Initialises a vector with a given number of cells to 0.0 */
    Vector(int numCells);

    /* Uses a given vector of doubles to initialise a Vector */
    Vector(std::vector<double> data);

    /* Retrives the number of cells in this vector */
    int getNumCells() const;

    /* Returns true if this is a column vector, false otherwise */
    bool isColVector();

    /* Sets a given value at a given position in the vector */
    void set(int i, double val);

    /* Retrives a value at a given position in the vector */
    double get(int i);

    /* Col Vector <=> Row Vector */
    void transpose();
};
}
}
