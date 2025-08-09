#ifndef DENSELAYER_H 
#define DENSELAYER_H

#include <layers/layer.hpp>
#include <hyperparameters.hpp>
#include <algorithm>

typedef Eigen::MatrixXd Matrix;

struct WeightBias {
	Matrix weights;
	Matrix biases;

	WeightBias(int inputDim, int outputDim, double initValue) {
		if (inputDim <= 0 || outputDim <= 0) return;
		weights = Matrix::Constant(inputDim, outputDim, initValue);
		biases = Matrix::Constant(1, outputDim, initValue);
	}
};

struct AdamVariables {
	WeightBias grads; // dl/dw, dl/db
	WeightBias m; // first moment
	WeightBias v; // second moment
	
	AdamVariables(int inputDim, int outputDim)
	: grads(inputDim, outputDim, 0),
	  m(inputDim, outputDim, 0),
	  v(inputDim, outputDim, 0) {}
};

class DenseLayer : public Layer {
	WeightBias parameters;
	AdamVariables adamVars;

public:
	DenseLayer(int inputDim, int outputDim) 
	: adamVars(inputDim, outputDim)
	{
		parameters.weights = Matrix::Random(inputDim, outputDim) * 0.01;
		parameters.biases = Matrix::Zero(1, outputDim);
	};

	Matrix FeedForward(Matrix& mat) {
		return (mat * parameters.weights) + parameters.biases;
	};

	Matrix FeedBackward(Matrix& mat, Matrix& grad) {
		adamVars.grads.weights = mat.transpose() * grad;
		adamVars.grads.biases = grad.colwise().sum();
		return grad * parameters.weights.transpose();
	};
};

#endif
