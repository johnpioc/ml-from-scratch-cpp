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

    /* Col Vector <=> Row Vector */
    void transpose();
};
}
}
