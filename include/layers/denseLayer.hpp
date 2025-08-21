#ifndef DENSELAYER_H 
#define DENSELAYER_H

#include "layers/layer.hpp"
#include "optimisers/optimiser.hpp"

struct WeightBias {
	Matrix weights;
	Matrix biases;

	WeightBias();
	WeightBias(int inputDim, int outputDim, double initValue);
};

class DenseLayer : public Layer {
	int inputDim, outputDim;
	WeightBias parameters;
	WeightBias gradients;

public:
	DenseLayer(int inputDim, int outputDim);

	Matrix FeedForward(Matrix& mat) override;
	Matrix FeedBackward(Matrix& mat, Matrix& grad) override;

	void Optimise(Optimiser& optimiser) override;

	void ZeroGradients() override;

	WeightBias& GetParameters() {
		return parameters;
	}

	WeightBias& GetGradients() {
		return gradients;
	}

	int GetInputDim() {
		return inputDim;
	}

	int GetOutputDim() {
		return outputDim;
	}
};

#endif
