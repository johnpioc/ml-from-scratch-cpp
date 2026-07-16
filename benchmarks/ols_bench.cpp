#include <benchmark/benchmark.h>

#include <armadillo>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/OLS.hpp>
#include <mlpack.hpp>
#include <utility>

#include "data_generation.hpp"
#include "mlpack/methods/linear_regression/linear_regression.hpp"

using jmll::benchmark::data_generation::makeLinearDataset;
using jmll::core::Matrix;
using jmll::core::Vector;

static void ols_10(benchmark::State& state) {
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

static void mlpack_ols_10(benchmark::State& state) {
    // generate linear dataset
    std::pair<Matrix, Vector> linearDataset = makeLinearDataset(10, 10);

    // initialise an armadillo matrix with transposed dimensions of the linear dataset feature
    // matrix (because mlpack assumes columns as observations and rows as features), then populate
    // it
    arma::mat data(linearDataset.first.getNumCols(), linearDataset.first.getNumRows());

    for (int r = 0; r < linearDataset.first.getNumCols(); r++) {
        for (int c = 0; c < linearDataset.first.getNumRows(); c++) {
            data(r, c) = linearDataset.first.get(c, r);
        }
    }

    // initialise an armadillo row vector for labels and populate it
    arma::rowvec labels(linearDataset.second.getNumCells());

    for (int i = 0; i < linearDataset.second.getNumCells(); i++) {
        labels(i) = linearDataset.second.get(i);
    }

    for (auto _ : state) {
        mlpack::LinearRegression model;
        model.Train(data, labels);
        benchmark::DoNotOptimize(model);
    }
}

static void ols_100(benchmark::State& state) {
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

static void mlpack_ols_100(benchmark::State& state) {
    // generate linear dataset
    std::pair<Matrix, Vector> linearDataset = makeLinearDataset(100, 100);

    // initialise an armadillo matrix with transposed dimensions of the linear dataset feature
    // matrix (because mlpack assumes columns as observations and rows as features), then populate
    // it
    arma::mat data(linearDataset.first.getNumCols(), linearDataset.first.getNumRows());

    for (int r = 0; r < linearDataset.first.getNumCols(); r++) {
        for (int c = 0; c < linearDataset.first.getNumRows(); c++) {
            data(r, c) = linearDataset.first.get(c, r);
        }
    }

    // initialise an armadillo row vector for labels and populate it
    arma::rowvec labels(linearDataset.second.getNumCells());

    for (int i = 0; i < linearDataset.second.getNumCells(); i++) {
        labels(i) = linearDataset.second.get(i);
    }

    for (auto _ : state) {
        mlpack::LinearRegression model;
        model.Train(data, labels);
        benchmark::DoNotOptimize(model);
    }
}

static void ols_1000(benchmark::State& state) {
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

static void mlpack_ols_1000(benchmark::State& state) {
    // generate linear dataset
    std::pair<Matrix, Vector> linearDataset = makeLinearDataset(1000, 1000);

    // initialise an armadillo matrix with transposed dimensions of the linear dataset feature
    // matrix (because mlpack assumes columns as observations and rows as features), then populate
    // it
    arma::mat data(linearDataset.first.getNumCols(), linearDataset.first.getNumRows());

    for (int r = 0; r < linearDataset.first.getNumCols(); r++) {
        for (int c = 0; c < linearDataset.first.getNumRows(); c++) {
            data(r, c) = linearDataset.first.get(c, r);
        }
    }

    // initialise an armadillo row vector for labels and populate it
    arma::rowvec labels(linearDataset.second.getNumCells());

    for (int i = 0; i < linearDataset.second.getNumCells(); i++) {
        labels(i) = linearDataset.second.get(i);
    }

    for (auto _ : state) {
        mlpack::LinearRegression model;
        model.Train(data, labels);
        benchmark::DoNotOptimize(model);
    }
}

BENCHMARK(ols_10)->Unit(benchmark::kMillisecond);
BENCHMARK(mlpack_ols_10)->Unit(benchmark::kMillisecond);

BENCHMARK(ols_100)->Unit(benchmark::kMillisecond);
BENCHMARK(mlpack_ols_100)->Unit(benchmark::kMillisecond);

BENCHMARK(ols_1000)->Unit(benchmark::kMillisecond);
BENCHMARK(mlpack_ols_1000)->Unit(benchmark::kMillisecond);
