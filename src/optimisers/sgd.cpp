#include "optimisers/sgd.hpp"

SGDOptimiser::SGDOptimiser(double stepSize)
        : stepSize(stepSize)
{}

void SGDOptimiser::Update(DenseLayer& denseLayer) {
    WeightBias& parameters = denseLayer.GetParameters();
    AdamVariables& vars = denseLayer.GetAdamVars();

    parameters.weights -= stepSize * vars.grads.weights;
    parameters.biases -= stepSize * vars.grads.biases;
}
