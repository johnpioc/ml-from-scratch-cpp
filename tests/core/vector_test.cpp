#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <jmll/core/vector.hpp>
#include <random>
#include <vector>

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

double getRandomNum() {
    std::uniform_real_distribution dist(-10.0, 10.0);
    return dist(gen);
}

int getRandomIndex() {
    std::uniform_int_distribution dist(0, VECTOR_DIM - 1);
    return dist(gen);
}

bool approxEqual(double a, double b, double epsilon) { return std::abs(a - b) < epsilon; }

bool isVectorIdentical(Vector& actual, VectorXd& expected) {
    bool actualIsColVector = actual.isColVector();
    bool expectedIsColVector = expected.cols() == 1;

    if (actualIsColVector != expectedIsColVector) {
        std::cout << "isVectorIdentical() failed because actual is "
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
        std::cout << "isVectorIdentical() failed because actual is "
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

void initialiseVectors(Vector& actual, VectorXd& expected) {
    for (int i = 0; i < VECTOR_DIM; i++) {
        double curr = getRandomNum();
        actual.set(i, curr);
        expected(i) = curr;
    }
}

// ===============================================================================================
// FUZZ TEST
// ===============================================================================================
TEST(Vector, FuzzTest) {
    Vector actual(VECTOR_DIM);
    VectorXd expected(VECTOR_DIM);

    for (int t = 0; t < NUM_ITERATIONS; t++) {
        int operation = getRandomOperation();

        // Reinitialise vectors every 10 tests and at the start
        if (t % 10 == 0) {
            initialiseVectors(actual, expected);
        }

        switch (operation) {
            case 0: {  // Get Number of Cells
                EXPECT_EQ(actual.getNumCells(), expected.size());
                break;
            }
            case 1: {  // Set
                double number = getRandomNum();
                int index = getRandomIndex();
                actual.set(index, number);
                expected(index) = number;
                break;
            }
            case 2: {  // Get Data
                std::vector actualData = actual.getData();
                for (int i = 0; i < actualData.size(); i++) {
                    EXPECT_TRUE(approxEqual(actualData[i], expected(i), 1e-2));
                }
                EXPECT_TRUE(actualData.size() == expected.size());
                break;
            }
            case 3: {  // Get Data by Indices
                int numOfIndices = getRandomIndex() + 1;
                std::vector<int> indices(numOfIndices);
                for (int i = 0; i < numOfIndices; i++) {
                    indices[i] = getRandomIndex();
                }

                std::vector<double> data = actual.getDataByIndices(indices);
                for (int i = 0; i < data.size(); i++) {
                    EXPECT_TRUE(approxEqual(data[i], expected(indices[i]), 1e-2));
                }
                EXPECT_TRUE(data.size() == indices.size());
                break;
            }
            case 4: {  // Vector Addition
                Vector operandA(VECTOR_DIM);
                VectorXd operandB(VECTOR_DIM);
                initialiseVectors(operandA, operandB);
                actual += operandA;
                expected += operandB;
                break;
            }
            case 5: {  // Scalar Addition
                double operand = getRandomNum();
                actual += operand;
                expected += Eigen::VectorXd::Constant(VECTOR_DIM, operand);
                break;
            }
            case 6: {  // Vector Subtraction
                Vector operandA(VECTOR_DIM);
                VectorXd operandB(VECTOR_DIM);
                initialiseVectors(operandA, operandB);
                actual -= operandA;
                expected -= operandB;
                break;
            }
            case 7: {  // Scalar Subtraction
                double operand = getRandomNum();
                actual -= operand;
                expected -= Eigen::VectorXd::Constant(VECTOR_DIM, operand);
                break;
            }
            case 8: {  // Dot Product
                Vector operandA(VECTOR_DIM);
                VectorXd operandB(VECTOR_DIM);
                initialiseVectors(operandA, operandB);
                double actualProduct = actual * operandA;
                double expectedProduct = expected.dot(operandB);
                EXPECT_TRUE(approxEqual(actualProduct, expectedProduct, 1e-2));
                break;
            }
            case 9: {  // Scalar Multiplication
                double operand = getRandomNum();
                actual *= operand;
                expected = expected.cwiseProduct(Eigen::VectorXd::Constant(VECTOR_DIM, operand));
                break;
            }
            case 10: {  // Scalar Division
                double operand = getRandomNum();
                actual /= operand;
                expected =
                    expected.array() / Eigen::VectorXd::Constant(VECTOR_DIM, operand).array();
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
