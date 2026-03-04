#include <jmll/core/vector.hpp>
#include <jmll/core/matrix.hpp>
#include <jmll/models/tuning/evaluation_metric.hpp>

using namespace jmll::core;
using namespace jmll::models::tuning;

// =============================================================================================== 
// LINEAR REGRESSION EVALUATION METRICS
// =============================================================================================== 
double RSquared::evaluate(core::Vector& yPred, core::Vector& yTrue, int numOfPredictors) {
    int n = yPred.numCells;

    // Calculate mean response
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += yTrue.get(i);
    }
    double yHat = sum / n;

    // Calculate tss and rss
    double tss = 0, rss = 0;
    for (int i = 0; i < n; i++) {
        tss += std::pow(yTrue.get(i) - yHat, 2.0);
        rss += std::pow(yTrue.get(i) - yPred.get(i), 2.0);
    }

    // Return R^2
    return 1 - (rss / tss);
}
