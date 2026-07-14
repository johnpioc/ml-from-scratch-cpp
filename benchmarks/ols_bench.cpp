#include <benchmark/benchmark.h>

#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/OLS.hpp>
#include <utility>

#include "data_generation.hpp"

using jmll::core::Matrix;
using jmll::core::Vector;

static void BENCHMARK_ols(benchmark::State& state) {
    using jmll::benchmark::data_generation::makeLinearDataset;
    using jmll::models::OLS;
    std::pair<Matrix, Vector> linearDataset = makeLinearDataset(500, 10);

    Matrix data = linearDataset.first;
    Vector labels = linearDataset.second;

    for (auto _ : state) {
        OLS model;
        model.fit(data, labels);
        benchmark::DoNotOptimize(model);
    }
}

BENCHMARK(BENCHMARK_ols)->Unit(benchmark::kMillisecond);
