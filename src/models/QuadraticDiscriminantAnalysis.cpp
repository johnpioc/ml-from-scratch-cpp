#include "../Matrix.h"
#include <iomanip>
#include "QuadraticDiscriminantAnalysis.h"

#define NUMBER_OF_CLASSES 2

QuadraticDiscriminantAnalysis::QuadraticDiscriminantAnalysis(int numOfPredictors)
{
    this->numOfPredictors_ = numOfPredictors;
    this->prior_ = Matrix(NUMBER_OF_CLASSES, 1);
    this->means_ = std::vector<Matrix>(NUMBER_OF_CLASSES, Matrix(1, this->numOfPredictors_));
    this->covariances_ = std::vector<Matrix>(NUMBER_OF_CLASSES);
}

void QuadraticDiscriminantAnalysis::train(Matrix& trainX, Matrix& trainY)
{
    // Split the data into vectors for each class
    std::vector<std::vector<std::vector<double>>> splitData(NUMBER_OF_CLASSES);

    for (int i = 0; i < trainX.getNumRows(); i++) {
        splitData[trainY.get(i, 0)].push_back({ trainX.get(i, 0), trainX.get(i, 1) });
    }

    std::vector<Matrix> data;
    for (int i = 0; i < NUMBER_OF_CLASSES; i++) {
        data.push_back(Matrix(splitData[i].size(), this->numOfPredictors_, splitData[i]));
    }

    // Get Prior Probabilities
    for (int k = 0; k < NUMBER_OF_CLASSES; k++) {
        int n = trainX.getNumRows();
        int n_k = data[k].getNumRows();
        this->prior_.put(k, 0, static_cast<double>(n_k) / n);
    }

    // Get Mean vectors for each class
    for (int k = 0; k < NUMBER_OF_CLASSES; k++) {
        int n_k = data[k].getNumRows();
        for (int j = 0; j < this->numOfPredictors_; j++) {
            double sum = 0;

            for (int i = 0; i < n_k; i++) {
                sum += data[k].get(i, j);
            }

            this->means_[k].put(0, j, sum / n_k);
        }
    }

    // Get Covariance Matrices for each Class
    for (int k = 0; k < NUMBER_OF_CLASSES; k++) {
        int n_k = data[k].getNumRows();
        Matrix meanMatrix(n_k, this->numOfPredictors_);
        for (int i = 0; i < n_k; i++) {
            for (int j = 0; j < this->numOfPredictors_; j++) {
                meanMatrix.put(i, j, this->means_[k].get(0, j));
            }
        }

        this->covariances_[k] = (data[k] - meanMatrix).transpose() * (data[k] - meanMatrix) 
            * (1 / static_cast<double>(n_k - 1));
    }
}

void QuadraticDiscriminantAnalysis::test(Matrix& testX, Matrix& testY)
{
    // Get Covariance Matrix Inverses
    std::vector<Matrix> inverseCovariances(NUMBER_OF_CLASSES, 
        Matrix(this->numOfPredictors_, this->numOfPredictors_));

    for (int k = 0; k < NUMBER_OF_CLASSES; k++) {
        Matrix sigma = this->covariances_[k];
        double determinant = sigma.get(0,0) * sigma.get(1,1) - sigma.get(0,1) * sigma.get(1,0);
        inverseCovariances[k].put(0,0, sigma.get(1,1));
        inverseCovariances[k].put(1,1, sigma.get(0,0));
        inverseCovariances[k].put(1,0, -sigma.get(1,0));
        inverseCovariances[k].put(0,1, -sigma.get(0,1));
        inverseCovariances[k] = inverseCovariances[k] * (static_cast<double>(1) / determinant);
    }

    int numIncorrect = 0;
    // Go through each observation and get estimates
    for (int i = 0; i < testX.getNumRows(); i++) {
        Matrix estimates(NUMBER_OF_CLASSES, 1);
        for (int k = 0; k < NUMBER_OF_CLASSES; k++) {
            Matrix x = testX.getRow(i);
            Matrix mu = this->means_[k].getRow(0);
            Matrix sigma = this->covariances_[k];
            Matrix inverseSigma = inverseCovariances[k];
            double determinant = sigma.get(0,0) * sigma.get(1,1) 
                - sigma.get(0,1) * sigma.get(1,0);

            double prob = (
                x * inverseSigma * x.transpose() * ( -1.0 / 2.0)
                + mu * inverseSigma * x.transpose() 
                - mu * inverseSigma * mu.transpose() * (1.0 / 2.0)
            ).get(0,0) - (1.0 / 2.0) * std::log(determinant) + std::log(this->prior_.get(k, 0));

            estimates.put(k, 0, prob);
        }

        double prediction = estimates.get(0,0) > estimates.get(1, 0) ? 0 : 1;
        if (prediction != testY.get(i, 0)) numIncorrect++;
    }
    
    // Compute Mean Error Rate
    double errorRate = static_cast<double>(numIncorrect) / testY.getNumRows() * 100;

    std::cout << "Implementation Error Rate: " << std::fixed << std::setprecision(2) <<
        errorRate << "\n";
}
