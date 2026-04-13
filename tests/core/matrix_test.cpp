#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <random>

#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>

// =============================================================================================== 
// CONSTANTS 
// =============================================================================================== 
std::mt19937 gen(0xC0FFEE);
constexpr int NUM_OPERATIONS = 14;
constexpr int NUM_ITERATIONS = 10000;
constexpr int MATRIX_DIM = 6;

// =============================================================================================== 
// HELPERS
// =============================================================================================== 
int getRandomOperation() {
    std::uniform_int_distribution dist(1, NUM_OPERATIONS);
    return dist(gen);
}

double getRandomNum() {
    std::uniform_real_distribution dist(-10.0, 10.0);
    return dist(gen);
}

void initialiseMatrices(jmll::core::Matrix& actual, Eigen::MatrixXd& expected) {
    for (int r = 0; r < MATRIX_DIM; r++) { 
        for (int c = 0; c < MATRIX_DIM; c++) {
            double curr = getRandomNum(); 
            actual.set(r, c, curr);
            expected(r, c) = curr;
        }
    }
}

int getRandomIndex() {
    std::uniform_int_distribution dist(0, MATRIX_DIM - 1);
    return dist(gen);
}

bool isStateIdentical(jmll::core::Matrix& actual, Eigen::MatrixXd& expected) {
    if (actual.numRows != expected.rows()) return false;
    if (actual.numCols != expected.cols()) return false;
    for (int r = 0; r < actual.numRows; r++) {
        for (int c = 0; c < actual.numCols; c++) {
            if (actual.get(r,c) != expected(r,c)) return false;
        }
    }

    return true;
}

// =============================================================================================== 
// FUZZ TEST
// =============================================================================================== 
TEST(Matrix, FuzzTest) {
    jmll::core::Matrix actual(MATRIX_DIM, MATRIX_DIM);
    Eigen::MatrixXd expected(MATRIX_DIM, MATRIX_DIM);

    initialiseMatrices(actual, expected);

    for (int t = 0; t < NUM_ITERATIONS; t++) {
        int operation = getRandomOperation();

        switch (operation) {
            case 0: { // Get
                int r = getRandomIndex();
                int c = getRandomIndex();
                EXPECT_EQ(actual.get(r,c), expected(r,c));
                break;
            }
            case 1: { // Get Row
                int r = getRandomIndex();
                jmll::core::Vector actualRow = actual.getRow(r);
                Eigen::VectorXd expectedRow = expected.row(r);

                for (int i = 0; i < expectedRow.size(); i++) {
                    EXPECT_EQ(actualRow.get(i), expectedRow(i));
                }
                break;
            }
            case 2: { // Get Col
                int c = getRandomIndex();
                jmll::core::Vector actualCol = actual.getCol(c);
                Eigen::VectorXd expectedCol = expected.col(c);

                for (int i = 0; i < expectedCol.size(); i++) {
                    EXPECT_EQ(actualCol.get(i), expectedCol(i));
                }
                break;
            }
            case 3: { // Get Rows
                int numRows = getRandomIndex() + 1;
                std::vector<int> indices;
                for (int i = 0; i < numRows; i++) indices.push_back(getRandomIndex());
                jmll::core::Matrix actualSliced = actual.getRows(indices);
                Eigen::MatrixXd expectedSliced = expected(indices, Eigen::all);
                EXPECT_TRUE(isStateIdentical(actualSliced, expectedSliced));
                break;
            }
            case 4: { // Set
                int r = getRandomIndex();
                int c = getRandomIndex();
                int value = getRandomNum();
                actual.set(r, c, value);
                expected(r,c) = value;
                break;
            }
            case 5: { // Matrix Multiplication
                jmll::core::Matrix operandA(actual.numRows, actual.numCols);
                Eigen::MatrixXd operandB(actual.numRows, actual.numCols);
                initialiseMatrices(operandA, operandB);
                actual = actual * operandA;
                expected = expected * operandB;
                break;
            }
            case 6: { // Matrix vector Multiplication
                jmll::core::Vector operandA(actual.numCols);
                Eigen::VectorXd operandB(actual.numCols);

                for (int i = 0; i < actual.numCols; i++) {
                    double curr = getRandomNum();
                    operandA.set(i, curr);
                    operandB(i) = curr;
                }

                jmll::core::Vector actualProduct = actual * operandA;
                Eigen::VectorXd expectedProduct = expected * operandB;

                for (int i = 0; i < actual.numCols; i++) {
                    EXPECT_EQ(actualProduct.get(i), expectedProduct(i));
                }
                break;
            }
            case 7: { // Matrix Scalar Multiplication
                double operand = getRandomNum();
                actual = actual * operand;
                expected = expected * operand;
                break;
            }
            case 8: { // Matrix Addition
                jmll::core::Matrix operandA(actual.numRows, actual.numCols);
                Eigen::MatrixXd operandB(actual.numRows, actual.numCols);
                initialiseMatrices(operandA, operandB);
                actual = actual + operandA;
                expected = expected + operandB;
                break;
            }
        }

        bool outcome = isStateIdentical(actual, expected);
        std::cout << "Operation: " << operation << std::endl;
        EXPECT_TRUE(outcome);
        if (!outcome) break;
    }
    // Transpose
    // Inverse
    // Get Column Means
    // Get Row Means
    // Get Column Variances
    // Get Row Variances

}

