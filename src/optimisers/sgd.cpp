#include "optimisers/sgd.hpp"

SGDOptimiser::SGDOptimiser(double stepSize)
        : stepSize(stepSize)
{}

void SGDOptimiser::Update(DenseLayer& denseLayer) {
    const WeightBias& grads = denseLayer.GetGradients();

    // constructor for WeightBias takes rvalues (weightUpdate, biasUpdate)
    WeightBias update {
        stepSize * grads.weights,
        stepSize * grads.biases
    };

    denseLayer.UpdateParameters(update);
}
