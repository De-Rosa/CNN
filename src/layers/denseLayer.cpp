#include "layers/denseLayer.hpp"
#include <stdexcept>

WeightBias::WeightBias() = default;

WeightBias::WeightBias(int inputDim, int outputDim, double initValue)
{
    if (inputDim <= 0 || outputDim <= 0) throw std::runtime_error("invalid dimensions");
    weights = Matrix::Constant(inputDim, outputDim, initValue);
    biases = Matrix::Constant(1, outputDim, initValue);
};

DenseLayer::DenseLayer(int inputDim, int outputDim) 
        : inputDim(inputDim)
        , outputDim(outputDim)
        , parameters(inputDim, outputDim, 0) 
{
    parameters.weights = Matrix::Random(inputDim, outputDim) * 0.01;
    parameters.biases = Matrix::Zero(1, outputDim);
};

Matrix DenseLayer::FeedForward(Matrix& mat) {
    Matrix z = mat * parameters.weights;
    Matrix biases_replicated = parameters.biases.replicate(mat.rows(), 1);

    return z + biases_replicated;
};

Matrix DenseLayer::FeedBackward(Matrix& mat, Matrix& grad) {
    gradients.weights = mat.transpose() * grad;
    gradients.biases = grad.colwise().sum();

    return grad * parameters.weights.transpose();
};

void DenseLayer::Optimise(Optimiser& optimiser) {
    optimiser.Update(*this);
};

void DenseLayer::ZeroGradients() {
    gradients.weights.setZero();
    gradients.biases.setZero();
};
