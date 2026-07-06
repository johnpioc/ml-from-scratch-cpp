#pragma once

#include <concepts>
#include <jmll/models/detail/model.hpp>
#include <utility>

namespace jmll::models::tuning {

template <detail::Model ModelType>
class GridSearcher {
   private:
    std::vector<int> getParamIndices(int index, std::vector<size_t>& paramSizes) {
        std::vector<int> paramIndices(paramSizes.size());

        int divisor = 1;
        for (int i = 0; i < paramSizes.size(); i++) {
            paramIndices[i] = (index / divisor) % paramSizes[i];
            divisor *= paramSizes[i];
        }

        return paramIndices;
    }

    template <size_t... Indexes, typename... ParamVector>
    auto getComboTuple(std::vector<int>& indices, std::index_sequence<Indexes...>,
                       std::vector<ParamVector>&... paramVectors) {
        return std::make_tuple(paramVectors[indices[Indexes]]...);
    }

   public:
    template <typename... Params>
        requires ::std::constructible_from<ModelType, Params...>
    ModelType get(core::Matrix& x, core::Vector& y, std::vector<Params>&... paramVectors) {
        std::vector<size_t> paramSizes = {paramVectors.size()...};
        int numParams = paramSizes.size();

        int totalCombinations = 1;
        for (int s : paramSizes) totalCombinations *= s;

        std::vector<int> paramIndices = getParamIndices(0, paramSizes);
        auto currentCombo = getComboTuple(
            paramIndices, std::make_index_sequence<sizeof...(Params)>{}, paramVectors...);

        ModelType output = std::apply(
            [](auto&&... args) { return ModelType(std::forward<decltype(args)>(args)...); },
            currentCombo);

        double bestScore = output.crossValidate(x, y, 4);

        for (int i = 1; i < totalCombinations; i++) {
            paramIndices = getParamIndices(i, paramSizes);
            currentCombo = getComboTuple(
                paramIndices, std::make_index_sequence<sizeof...(Params)>{}, paramVectors...);

            ModelType model = std::apply(
                [](auto&&... args) {
                    return ModelType(std::forward<decltype(args)>(args)...);
                },
                currentCombo);

            double score = model.crossValidate(x, y, 4);

            if (score > bestScore) {
                bestScore = score;
                output = model;
            }
        }

        return output;
    }
};

}  // namespace jmll::models::tuning
