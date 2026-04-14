#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <random>

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

bool isMatrixIdentical(jmll::core::Matrix& actual, Eigen::MatrixXd& expected) {
    if (actual.getNumRows() != expected.rows()) return false;
    if (actual.getNumCols() != expected.cols()) return false;
    for (int r = 0; r < actual.getNumRows(); r++) {
        for (int c = 0; c < actual.getNumCols(); c++) {
            if (actual.get(r, c) != expected(r, c)) return false;
        }
    }

    return true;
}

bool isVectorIdentical(jmll::core::Vector& actual, Eigen::VectorXd& expected) {
    if (expected.rows() != (actual.isColVector() ? actual.getNumCells() : 1)) return false;
    if (expected.cols() != (actual.isColVector() ? 1 : actual.getNumCells())) return false;

    for (int i = 0; i < actual.getNumCells(); i++) {
        if (actual.get(i) != expected(i)) return false;
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
            case 0: {  // Get
                int r = getRandomIndex();
                int c = getRandomIndex();
                EXPECT_EQ(actual.get(r, c), expected(r, c));
                break;
            }
            case 1: {  // Get Row
                int r = getRandomIndex();
                jmll::core::Vector actualRow = actual.getRow(r);
                Eigen::VectorXd expectedRow = expected.row(r);

                EXPECT_TRUE(isVectorIdentical(actualRow, expectedRow));
                break;
            }
            case 2: {  // Get Col
                int c = getRandomIndex();
                jmll::core::Vector actualCol = actual.getCol(c);
                Eigen::VectorXd expectedCol = expected.col(c);

                EXPECT_TRUE(isVectorIdentical(actualCol, expectedCol));
                break;
            }
            case 3: {  // Get Rows
                int numRows = getRandomIndex() + 1;
                std::vector<int> indices;
                for (int i = 0; i < numRows; i++) indices.push_back(getRandomIndex());
                jmll::core::Matrix actualSliced = actual.getRows(indices);
                Eigen::MatrixXd expectedSliced = expected(indices, Eigen::all);
                EXPECT_TRUE(isMatrixIdentical(actualSliced, expectedSliced));
                break;
            }
            case 4: {  // Set
                int r = getRandomIndex();
                int c = getRandomIndex();
                int value = getRandomNum();
                actual.set(r, c, value);
                expected(r, c) = value;
                break;
            }
            case 5: {  // Matrix Multiplication
                jmll::core::Matrix operandA(actual.getNumRows(), actual.getNumCols());
                Eigen::MatrixXd operandB(actual.getNumRows(), actual.getNumCols());
                initialiseMatrices(operandA, operandB);
                actual = actual * operandA;
                expected = expected * operandB;
                break;
            }
            case 6: {  // Matrix vector Multiplication
                jmll::core::Vector operandA(actual.getNumCols());
                Eigen::VectorXd operandB(actual.getNumCols());

                for (int i = 0; i < actual.getNumCols(); i++) {
                    double curr = getRandomNum();
                    operandA.set(i, curr);
                    operandB(i) = curr;
                }

                jmll::core::Vector actualProduct = actual * operandA;
                Eigen::VectorXd expectedProduct = expected * operandB;

                EXPECT_TRUE(isVectorIdentical(actualProduct, expectedProduct));
                break;
            }
            case 7: {  // Matrix Scalar Multiplication
                double operand = getRandomNum();
                actual = actual * operand;
                expected = expected * operand;
                break;
            }
            case 8: {  // Matrix Addition
                jmll::core::Matrix operandA(actual.getNumRows(), actual.getNumCols());
                Eigen::MatrixXd operandB(actual.getNumRows(), actual.getNumCols());
                initialiseMatrices(operandA, operandB);
                actual = actual + operandA;
                expected = expected + operandB;
                break;
            }
            case 9: {  // Transpose
                actual = actual.transpose();
                expected = expected.transpose();
                break;
            }
            case 10: {  // Inverse
                actual = actual.inverse();
                expected = expected.inverse();
                break;
            }
            case 11: {  // Get Column Means
                jmll::core::Vector actualColMeans = actual.getColMeans();
                Eigen::VectorXd expectedColMeans = expected.colwise().mean();
                EXPECT_TRUE(isVectorIdentical(actualColMeans, expectedColMeans));
                break;
            }
            case 12: {  // Get Row Means
                jmll::core::Vector actualRowMeans = actual.getRowMeans();
                Eigen::VectorXd expectedRowMeans = expected.rowwise().mean();
                EXPECT_TRUE(isVectorIdentical(actualRowMeans, expectedRowMeans));
                break;
            }
            case 13: {  // Get Column Variances
                jmll::core::Vector actualColVariances = actual.getColVariances();
                Eigen::VectorXd expectedColVariances =
                    ((expected.rowwise() - expected.colwise().mean())
                         .array()
                         .square()
                         .colwise()
                         .sum()) /
                    (expected.rows() - 1);
                EXPECT_TRUE(isVectorIdentical(actualColVariances, expectedColVariances));
                break;
            }
            case 14: {  // Get Row Variances
                jmll::core::Vector actualRowVariances = actual.getRowVariances();
                Eigen::VectorXd expectedRowVariances =
                    ((expected.colwise() - expected.rowwise().mean())
                         .array()
                         .square()
                         .rowwise()
                         .sum()) /
                    (expected.cols() - 1);
                EXPECT_TRUE(isVectorIdentical(actualRowVariances, expectedRowVariances));
                break;
            }
        }

        bool outcome = isMatrixIdentical(actual, expected);
        EXPECT_TRUE(outcome);
        if (!outcome) break;
    }
}
