#include <iomanip>
#include <stdexcept>

#include "../Matrix.h"
#include "LinearRegression.h"

LinearRegression::LinearRegression(size_t numOfPredictors)
{
    this->numOfPredictors_ = numOfPredictors;
    this->coefficient_ = 0;
    this->intercept_ = 0;
}

void LinearRegression::train(Matrix& trainX, Matrix& trainY)
{
    // First find sample x mean and sample y mean
    double xSum = 0;
    double ySum = 0;

    for (int i = 0; i < trainX.getNumRows(); i++) {
        xSum += trainX.get(i, 0);
        ySum += trainY.get(i, 0);
    }

    double xMean = xSum / trainX.getNumRows();
    double yMean = ySum / trainY.getNumRows();

    // Get Coefficient estimate
    double numerator = 0;
    double denominator = 0;

    for (int i = 0; i < trainX.getNumRows(); i++) {
        numerator += (trainX.get(i,0) - xMean) * (trainY.get(i, 0) - yMean);
        denominator += std::pow(trainX.get(i, 0) - xMean, 2);
    }

    this->coefficient_ = numerator / denominator;

    // Get intercept estimate
    this->intercept_ = yMean - this->coefficient_ * xMean;
}

void LinearRegression::test(Matrix& testX, Matrix& testY)
{
    // Find sample x mean and sample y mean
    double xSum = 0;
    double ySum = 0;

    for (int i = 0; i < testX.getNumRows(); i++) {
        xSum += testX.get(i,0);
        ySum += testY.get(i,0);
    }

    double xMean = xSum / testX.getNumRows();
    double yMean = ySum / testY.getNumRows();

    // Get TSS and RSS
    double tss = 0;
    double rss = 0;

    for (int i = 0; i < testX.getNumRows(); i++) {
        double estimate = this->intercept_ + this->coefficient_ * testX.get(i, 0); 
        tss += std::pow(testY.get(i, 0) - yMean, 2);
        rss += std::pow(estimate - testY.get(i,0), 2);
    }

    // Compute R Squared
    double rSquared = 1 - rss / tss;

    std::cout << "Implementation R Squared: " << std::fixed << std::setprecision(2) << rSquared 
        << '\n';

}
