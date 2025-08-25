#ifndef DENSELAYER_H 
#define DENSELAYER_H

#include "layers/layer.hpp"
#include "optimisers/optimiser.hpp"

struct WeightBias {
	Matrix weights;
	Matrix biases;

	WeightBias();
	WeightBias(Matrix&& weights, Matrix&& biases);
	WeightBias(int inputDim, int outputDim, double initValue);
};

class DenseLayer : public Layer {
public:
	DenseLayer(int inputDim, int outputDim);

	Matrix FeedForward(const Matrix& mat) const override;
	Matrix FeedBackward(const Matrix& mat, const Matrix& grad) override;

	void Optimise(Optimiser& optimiser) override;

	void ZeroGradients() override;
	
	void UpdateParameters(const WeightBias& update);

	const WeightBias& GetGradients() const {
		return gradients;
	}

	int GetInputDim() const {
		return inputDim;
	}

	int GetOutputDim() const {
		return outputDim;
	}

private:
	int inputDim, outputDim;
	WeightBias parameters;
	WeightBias gradients;
};

#endif
