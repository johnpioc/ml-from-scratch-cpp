#pragma once

#include <mlfs/models/tuning/evaluation_metric.hpp>
#include <mlfs/models/detail/model.hpp>
#include <mlfs/core/matrix.hpp>
#include <mlfs/core/vector.hpp>

namespace mlfs::models::detail {

template <tuning::EvaluationMetric EvaluationMetric>
class CrossValidator {
private:
    EvaluationMetric metric_;

public:
    template <Model Model, typename... Args>
    double crossValidate(Model model, core::Matrix& x, core::Vector& y, int numOfFolds, 
        Args... args) {
        int numPerFold = x.numRows / numOfFolds;
        
        double sum = 0.0;
        for (int i = 0; i < numOfFolds; i++) {
            // Retrieve train and test indices
            std::vector<int> trainIndices;
            std::vector<int> testIndices;

            int start = i * numPerFold;
            int end = start + numPerFold;
            for (int j = 0; j < x.numRows; j++) {
                if (start <= j && j <= end) testIndices.push_back(j);
                else trainIndices.push_back(j);
            }

            // Retrieve train and test data
            core::Matrix xTrain(x.getRows(trainIndices));
            core::Vector yTrain(y.getDataByIndices(trainIndices));
            core::Matrix xTest(x.getRows(testIndices));
            core::Vector yTest(y.getDataByIndices(testIndices));

            model.fit(xTrain, yTrain);
            core::Vector yPred = model.predict(xTest);
            sum += model.evaluate(yPred, yTest, std::forward<Args>(args)...);
        }

        return sum / (double) numOfFolds;
    }
};
}
