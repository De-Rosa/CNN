#include "optimisers/sgd.hpp"

SGDOptimiser::SGDOptimiser(double stepSize)
        : stepSize(stepSize)
{}

void SGDOptimiser::Update(DenseLayer& denseLayer) {
    WeightBias& parameters = denseLayer.GetParameters();
    WeightBias& grads = denseLayer.GetGradients();

    parameters.weights -= stepSize * grads.weights;
    parameters.biases -= stepSize * grads.biases;
}
