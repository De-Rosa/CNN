#include "layers/adamLayer.hpp"

WeightBias::WeightBias() = default;

WeightBias::WeightBias(int inputDim, int outputDim, double initValue) {
    if (inputDim <= 0 || outputDim <= 0) throw std::runtime_error("invalid dimensions");
    weights = Matrix::Constant(inputDim, outputDim, initValue);
    biases = Matrix::Constant(1, outputDim, initValue);
}

AdamVariables::AdamVariables(int inputDim, int outputDim)
	: grads(inputDim, outputDim, 0)
	, m(inputDim, outputDim, 0)
	, v(inputDim, outputDim, 0)
{}

AdamLayer::AdamLayer(int inputDim, int outputDim)
	: adamVars(inputDim, outputDim)
{}


