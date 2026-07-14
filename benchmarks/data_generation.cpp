#include "data_generation.hpp"

#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <random>
#include <utility>
#include <vector>

// Random number generator parameters
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> randomRealDist(-10.0, 10.0);

// ===============================================================================================
// HELPERS
// ===============================================================================================
double generateRandomNum() { return randomRealDist(gen); }

void addGuassianNoise(jmll::core::Vector& vec) {
    std::normal_distribution<double> noiseDist(0, generateRandomNum());

    for (int i = 0; i < vec.getNumCells(); i++) {
        vec.set(i, vec.get(i) + noiseDist(gen));
    }
}

// ===============================================================================================
// DATA GENERATION METHODS
// ===============================================================================================
namespace jmll::benchmark::data_generation {
using jmll::core::Matrix;
using jmll::core::Vector;

std::pair<Matrix, Vector> makeLinearDataset(int n, int d) {
    // Generate d random slopes and a random intercept
    std::vector<double> slopeVector(d, 0.0);
    for (int i = 0; i < d; i++) {
        slopeVector[i] = generateRandomNum();
    }
    Vector slope(slopeVector);
    double intercept = generateRandomNum();

    // Generate n datapoints and labels
    Matrix data(n, d);
    Vector labels(n);

    for (int r = 0; r < n; r++) {
        for (int c = 0; c < d; c++) {
            data.set(r, c, generateRandomNum());
        }
    }

    for (int i = 0; i < n; i++) {
        labels.set(i, data.getRow(i) * slope + intercept);
    }

    // Add guassian noise to labels
    addGuassianNoise(labels);

    return {data, labels};
}
}  // namespace jmll::benchmark::data_generation
