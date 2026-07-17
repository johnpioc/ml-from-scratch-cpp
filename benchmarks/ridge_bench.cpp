#include <benchmark/benchmark.h>

#include <armadillo>
#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <jmll/models/ridge.hpp>
#include <mlpack.hpp>
#include <random>
#include <utility>

#include "data_generation.hpp"
#include "mlpack/methods/linear_regression/linear_regression.hpp"

using jmll::benchmark::data_generation::makeLinearDataset;
using jmll::core::Matrix;
using jmll::core::Vector;

std::random_device rd_;
std::mt19937 gen_(rd_());
std::uniform_real_distribution<double> dist_(0, 10);

double getRandomLambda() { return dist_(gen_); }

static void ridge_10(benchmark::State& state) {
    using jmll::models::Ridge;
    std::pair<Matrix, Vector> linearDataset = makeLinearDataset(10, 10);

    Matrix data = linearDataset.first;
    Vector labels = linearDataset.second;

    for (auto _ : state) {
        Ridge model(getRandomLambda());
        model.fit(data, labels);
        benchmark::DoNotOptimize(model);
    }
}

static void mlpack_ridge_10(benchmark::State& state) {
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
        model.Train(data, labels, getRandomLambda());
        benchmark::DoNotOptimize(model);
    }
}

static void ridge_100(benchmark::State& state) {
    using jmll::models::Ridge;
    std::pair<Matrix, Vector> linearDataset = makeLinearDataset(100, 100);

    Matrix data = linearDataset.first;
    Vector labels = linearDataset.second;

    for (auto _ : state) {
        Ridge model(getRandomLambda());
        model.fit(data, labels);
        benchmark::DoNotOptimize(model);
    }
}

static void mlpack_ridge_100(benchmark::State& state) {
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
        model.Train(data, labels, getRandomLambda());
        benchmark::DoNotOptimize(model);
    }
}

static void ridge_1000(benchmark::State& state) {
    using jmll::models::Ridge;
    std::pair<Matrix, Vector> linearDataset = makeLinearDataset(1000, 1000);

    Matrix data = linearDataset.first;
    Vector labels = linearDataset.second;

    for (auto _ : state) {
        Ridge model(getRandomLambda());
        model.fit(data, labels);
        benchmark::DoNotOptimize(model);
    }
}

static void mlpack_ridge_1000(benchmark::State& state) {
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
        model.Train(data, labels, getRandomLambda());
        benchmark::DoNotOptimize(model);
    }
}

BENCHMARK(ridge_10)->Unit(benchmark::kMillisecond);
BENCHMARK(mlpack_ridge_10)->Unit(benchmark::kMillisecond);

BENCHMARK(ridge_100)->Unit(benchmark::kMillisecond);
BENCHMARK(mlpack_ridge_100)->Unit(benchmark::kMillisecond);

BENCHMARK(ridge_1000)->Unit(benchmark::kMillisecond);
BENCHMARK(mlpack_ridge_1000)->Unit(benchmark::kMillisecond);
