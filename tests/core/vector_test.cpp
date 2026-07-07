#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <jmll/core/vector.hpp>
#include <random>

using Eigen::RowVectorXd;
using Eigen::VectorXd;
using jmll::core::Vector;

// ===============================================================================================
// CONSTANTS
// ===============================================================================================
std::mt19937 gen(0xC0FFEE);
constexpr int NUM_OPERATIONS = 10;
constexpr int NUM_ITERATIONS = 10000;
constexpr int VECTOR_DIM = 10;

// ===============================================================================================
// HELPERS
// ===============================================================================================
int getRandomOperation() {
    std::uniform_int_distribution dist(0, NUM_OPERATIONS);
    return dist(gen);
}

bool approxEqual(double a, double b, double epsilon) { return std::abs(a - b) < epsilon; }

bool isVectorIdentical(Vector& actual, VectorXd& expected) {
    bool actualIsColVector = actual.isColVector();
    bool expectedIsColVector = expected.cols() == 1;

    if (actualIsColVector != expectedIsColVector) {
        std::cout << "isVectorIdentical() failed because actual is"
                  << (actualIsColVector ? "a col vector" : "a row vector") << " and expected is "
                  << (expectedIsColVector ? "a col vector" : "a row vector") << "\n";
        return false;
    }

    for (int i = 0; i < actual.getNumCells(); i++) {
        if (!approxEqual(actual.get(i), expected(i), 1e-2)) {
            std::cout << "isVectorIdentical() failed because cells don't equal each other\n";
            return false;
        }
    }

    return true;
}

bool isVectorIdentical(Vector& actual, RowVectorXd& expected) {
    bool actualIsColVector = actual.isColVector();
    bool expectedIsColVector = expected.cols() == 1;

    if (actualIsColVector != expectedIsColVector) {
        std::cout << "isVectorIdentical() failed because actual is"
                  << (actualIsColVector ? "a col vector" : "a row vector") << " and expected is "
                  << (expectedIsColVector ? "a col vector" : "a row vector") << "\n";
        return false;
    }

    for (int i = 0; i < actual.getNumCells(); i++) {
        if (!approxEqual(actual.get(i), expected(i), 1e-2)) {
            std::cout << "isVectorIdentical() failed because cells don't equal each other\n";
            return false;
        }
    }

    return true;
}

// ===============================================================================================
// FUZZ TEST
// ===============================================================================================
TEST(Vector, FuzzTest) {
    Vector actual(VECTOR_DIM);
    VectorXd expected(VECTOR_DIM);

    for (int t = 0; t < NUM_ITERATIONS; t++) {
        int operation = getRandomOperation();

        switch (operation) {
            case 0: {  //
                break;
            }
        }

        bool outcome = isVectorIdentical(actual, expected);
        EXPECT_TRUE(outcome);
        if (!outcome) {
            std::cout << "Test failed on operation: " << operation << "\n";
            break;
        }
    }
}
