#include "layers/denseLayer.hpp"
#include <stdexcept>

DenseLayer::DenseLayer(int inputDim, int outputDim) 
        : AdamLayer(inputDim, outputDim)
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
    adamVars.grads.weights = mat.transpose() * grad;
    adamVars.grads.biases = grad.colwise().sum();

    return grad * parameters.weights.transpose();
};

void DenseLayer::Optimise(Optimiser& optimiser) {
    optimiser.Update(*this);
}

void DenseLayer::ZeroGradients() {
    adamVars.grads.weights.setZero();
    adamVars.grads.biases.setZero();
}
