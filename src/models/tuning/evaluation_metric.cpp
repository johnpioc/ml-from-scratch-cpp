#include <cmath>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/tuning/evaluation_metric.hpp>

using namespace jmll::core;
using namespace jmll::models::tuning;

// ===============================================================================================
// HELPERS
// ===============================================================================================
double calculateRSS(Vector& yPred, Vector& yTrue) {
    int n = yPred.numCells;
    double rss = 0;

    for (int i = 0; i < n; i++) {
        rss += std::pow(yTrue.get(i) - yPred.get(i), 2.0);
    }

    return rss;
}

// ===============================================================================================
// REGRESSION EVALUATION METRICS
// ===============================================================================================
double RSquared::evaluate(Vector& yPred, Vector& yTrue, int numOfPredictors) {
    int n = yPred.numCells;

    // Calculate mean response
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += yTrue.get(i);
    }
    double yHat = sum / n;

    // Calculate tss and rss
    double tss = 0, rss = calculateRSS(yPred, yTrue);
    for (int i = 0; i < n; i++) {
        tss += std::pow(yTrue.get(i) - yHat, 2.0);
    }

    // Return R^2
    return 1 - (rss / tss);
}

double MallowsCp::evaluate(Vector& yPred, Vector& yTrue, int numOfPredictors) {
    int n = yPred.numCells;
    double rss = calculateRSS(yPred, yTrue);
    double varianceHat = rss / (n - numOfPredictors - 1);
    return (1 / (double)n) * (rss + 2 * numOfPredictors * varianceHat);
}

double AIC::evaluate(Vector& yPred, Vector& yTrue, int numOfPredictors) {
    int n = yPred.numCells;
    double rss = calculateRSS(yPred, yTrue);
    return n * std::log(rss / n) + 2 * numOfPredictors;
}

double BIC::evaluate(Vector& yPred, Vector& yTrue, int numOfPredictors) {
    int n = yPred.numCells;
    double rss = calculateRSS(yPred, yTrue);
    double varianceHat = rss / (n - numOfPredictors - 1);
    return (1 / (double)n) * (rss + std::log(n) * numOfPredictors * varianceHat);
}

double AdjustedRSquared::evaluate(Vector& yPred, Vector& yTrue, int numOfPredictors) {
    int n = yPred.numCells;

    // Calculate mean response
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += yTrue.get(i);
    }
    double yHat = sum / n;

    // Calculate tss and rss
    double tss = 0, rss = calculateRSS(yPred, yTrue);
    for (int i = 0; i < n; i++) {
        tss += std::pow(yTrue.get(i) - yHat, 2.0);
    }

    return (rss / (n - numOfPredictors - 1)) / (tss / (n - 1));
}
