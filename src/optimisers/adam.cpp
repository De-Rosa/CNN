#include "optimisers/adam.hpp"

#include <cmath>

AdamState::AdamState(int inputDim, int outputDim)
	: m(inputDim, outputDim, 0)
	, v(inputDim, outputDim, 0)
{}

AdamOptimiser::AdamOptimiser(double decay1, double decay2, double stepSize, double epsilon)
        : decay1(decay1), decay2(decay2), stepSize(stepSize), epsilon(epsilon)
{}

AdamState& AdamOptimiser::GetState(DenseLayer& denseLayer) {
    // https://stackoverflow.com/a/72808071
    return states.try_emplace(&denseLayer, denseLayer.GetInputDim(), denseLayer.GetOutputDim()).first->second;
};

// https://optimization.cbe.cornell.edu/index.php?title=Adam
void AdamOptimiser::Update(DenseLayer& denseLayer) {
    const WeightBias& grads = denseLayer.GetGradients();

    AdamState& state = GetState(denseLayer);

    ++time;

    // momentum
    state.m.weights = decay1 * state.m.weights +
            (1.0 - decay1) * grads.weights;
    state.m.biases  = decay1 * state.m.biases +
            (1.0 - decay1) * grads.biases;

    // rms
    state.v.weights = decay2 * state.v.weights +
            (1.0 - decay2) * grads.weights.array().square().matrix();
    state.v.biases  = decay2 * state.v.biases +
            (1.0 - decay2) * grads.biases.array().square().matrix();

    // bias correction
    Matrix corr_mw = state.m.weights / (1.0 - std::pow(decay1, time));
    Matrix corr_mb = state.m.biases  / (1.0 - std::pow(decay1, time));

    Matrix corr_vw = state.v.weights / (1.0 - std::pow(decay2, time));
    Matrix corr_vb = state.v.biases  / (1.0 - std::pow(decay2, time));

    // constructor for WeightBias takes rvalues: (weightUpdate, biasUpdate)
    WeightBias update {
        stepSize * (corr_mw.array() / (corr_vw.array().sqrt() + epsilon)).matrix(),
        stepSize * (corr_mb.array() / (corr_vb.array().sqrt() + epsilon)).matrix()
    };

    denseLayer.UpdateParameters(update);
};

