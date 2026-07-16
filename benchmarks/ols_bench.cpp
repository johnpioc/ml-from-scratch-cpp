#include <benchmark/benchmark.h>

#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/OLS.hpp>
#include <mlpack.hpp>
#include <utility>

#include "data_generation.hpp"

using jmll::core::Matrix;
using jmll::core::Vector;

static void ols_10(benchmark::State& state) {
    using jmll::benchmark::data_generation::makeLinearDataset;
    using jmll::models::OLS;
    std::pair<Matrix, Vector> linearDataset = makeLinearDataset(10, 10);

    Matrix data = linearDataset.first;
    Vector labels = linearDataset.second;

    for (auto _ : state) {
        OLS model;
        model.fit(data, labels);
        benchmark::DoNotOptimize(model);
    }
}

static void ols_100(benchmark::State& state) {
    using jmll::benchmark::data_generation::makeLinearDataset;
    using jmll::models::OLS;
    std::pair<Matrix, Vector> linearDataset = makeLinearDataset(100, 100);

    Matrix data = linearDataset.first;
    Vector labels = linearDataset.second;

    for (auto _ : state) {
        OLS model;
        model.fit(data, labels);
        benchmark::DoNotOptimize(model);
    }
}

static void ols_1000(benchmark::State& state) {
    using jmll::benchmark::data_generation::makeLinearDataset;
    using jmll::models::OLS;
    std::pair<Matrix, Vector> linearDataset = makeLinearDataset(1000, 1000);

    Matrix data = linearDataset.first;
    Vector labels = linearDataset.second;

    for (auto _ : state) {
        OLS model;
        model.fit(data, labels);
        benchmark::DoNotOptimize(model);
    }
}

BENCHMARK(ols_10)->Unit(benchmark::kMillisecond);
BENCHMARK(ols_100)->Unit(benchmark::kMillisecond);
BENCHMARK(ols_1000)->Unit(benchmark::kMillisecond);
