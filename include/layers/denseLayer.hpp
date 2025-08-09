#ifndef DENSELAYER_H 
#define DENSELAYER_H

#include <layers/adamLayer.hpp>
#include <optimisers/optimiser.hpp>
#include <stdexcept>

class DenseLayer : public AdamLayer {
	WeightBias parameters;
public:
	DenseLayer(int inputDim, int outputDim) 
	: AdamLayer(inputDim, outputDim),
	  parameters(inputDim, outputDim, 0)
	{
		parameters.weights = Matrix::Random(inputDim, outputDim) * 0.01;
		parameters.biases = Matrix::Zero(1, outputDim);
	};

	Matrix FeedForward(Matrix& mat) override {
		Matrix z = mat * parameters.weights;
		Matrix biases_replicated = parameters.biases.replicate(mat.rows(), 1);

		return z + biases_replicated;
	};

	Matrix FeedBackward(Matrix& mat, Matrix& grad) override {
		adamVars.grads.weights = mat.transpose() * grad;
		adamVars.grads.biases = grad.colwise().sum();
		return grad * parameters.weights.transpose();
	};

	void ZeroGradients() override {
		adamVars.grads.weights.setZero();
		adamVars.grads.biases.setZero();
	}

	void Optimise(Optimiser& optimiser) override {
		optimiser.Update(*this);
	}

	WeightBias& GetParameters() {
		return parameters;
	}
};

#endif
