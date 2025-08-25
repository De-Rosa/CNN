#include "layers/denseLayer.hpp"

#include <stdexcept>

WeightBias::WeightBias() = default;

WeightBias::WeightBias(Matrix&& w, Matrix&& b)
        : weights(std::move(w))
        , biases(std::move(b))
{
    if (weights.cols() != biases.cols() || biases.rows() != 1) throw std::runtime_error("initialising WeightBias with invalid matrix sizes");
};

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

Matrix DenseLayer::FeedForward(const Matrix& mat) const {
    Matrix z = mat * parameters.weights;

    // https://stackoverflow.com/questions/35280290/replicate-a-column-vectorxd-in-order-to-construct-a-matrixxd-in-eigen-c
    // z is of size (samples, outputDim) so we need to replicate across the samples dimension for addition
    Matrix replicatedBiases = parameters.biases.replicate(mat.rows(), 1);

    return z + replicatedBiases;
};

Matrix DenseLayer::FeedBackward(const Matrix& mat, const Matrix& grad) {
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

void DenseLayer::UpdateParameters(const WeightBias& update) {
    parameters.weights -= update.weights;
    parameters.biases -= update.biases;
}
