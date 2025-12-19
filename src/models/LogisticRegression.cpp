#include "LogisticRegression.h"
#include <iomanip>

#define LEARNING_RATE 0.000001

LogisticRegression::LogisticRegression()
{
    this->intercept_ = 0;
    this->coefficient_ = 1;
}

void LogisticRegression::train(Matrix& trainX, Matrix& trainY)
{
    // Estimate coefficients using gradient descent
    for (int i = 0; i < trainX.getNumRows(); i++) {
        double exponential = std::exp(this->intercept_ + this->coefficient_ * trainX.get(i,0));
        double estimate = exponential / (1 + exponential);
        double error = (trainY.get(i, 0) - estimate);
        double derivIntercept = error * trainX.get(i, 0);

        this->coefficient_-= LEARNING_RATE * derivIntercept;
        this->intercept_ -= LEARNING_RATE * error;
    }
}

void LogisticRegression::test(Matrix& testX, Matrix& testY)
{
    int numOfErrors = 0;

    for (int i = 0; i < testX.getNumRows(); i++) {
        double exponential = std::exp(this->intercept_ + this->coefficient_ * testX.get(i,0));
        double estimate = exponential / (1 + exponential);
        int prediction = estimate > 0.5 ? 1 : 0;
        if (prediction != testY.get(i, 0)) {
            numOfErrors++;
        }
    }

    double errorRate = static_cast<double>(numOfErrors) / testX.getNumRows() * 100;

    std::cout << "Implementation Error Rate: " << std::fixed << std::setprecision(2) << errorRate 
        << "%\n";
}
