#include <iomanip>
#include <stdexcept>

#include "../Matrix.h"
#include "LinearRegression.h"

LinearRegression::LinearRegression(size_t numOfPredictors)
{
    this->numOfPredictors_ = numOfPredictors;
    this->coefficients_ = Matrix(this->numOfPredictors_, 1);
    this->intercept_ = 0;
}

void LinearRegression::train(Matrix& trainX, Matrix& trainY)
{
    Matrix temp(this->numOfPredictors_, this->numOfPredictors_);
    bool inverseExists = (trainX.transpose() * trainX).inverse(temp);
    if (!inverseExists)
        throw std::runtime_error("Inverse does not exist for training data");

    this->coefficients_ = temp * trainX.transpose() * trainY;
    this->intercept_ = 0;
}

void LinearRegression::test(Matrix& testX, Matrix& testY)
{
    // Obtain Predictions
    int n = testY.getNumRows();
    Matrix predictions(n, 1);

    for (int i = 0; i < n; i++) {
        double prediction = this->intercept_;
        Matrix currentRow = testX.getRow(i);
        for (int j = 0; j < this->numOfPredictors_; j++) {
            prediction += this->coefficients_.get(j, 0) * currentRow.get(0, j);
        }
        predictions.put(i, 0, prediction);
    }

    // Get label mean
    double labelSum = 0;
    for (int i = 0; i < n; i++) {
        labelSum += testY.get(i, 0);
    }

    double labelMean = labelSum / n;

    // Compute RSS and TSS
    double rss = 0;
    double tss = 0;
    for (int i = 0; i < n; i++) {
        rss += std::pow(testY.get(i, 0) - predictions.get(i, 0),2);
        tss += std::pow(testY.get(i, 0) - labelMean, 2);
    }

    double r_squared = 1 - (rss / tss);

    std::cout << "Implementation R Squared: " << std::fixed << std::setprecision(2) 
        << r_squared << "\n";
}
