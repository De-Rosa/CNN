#ifndef DENSELAYER_H 
#define DENSELAYER_H

#include <layers/adamLayer.hpp>
#include <optimisers/optimiser.hpp>

class DenseLayer : public AdamLayer {
	WeightBias parameters;
public:
	DenseLayer(int inputDim, int outputDim);

	Matrix FeedForward(Matrix& mat) override;
	Matrix FeedBackward(Matrix& mat, Matrix& grad) override;

	void Optimise(Optimiser& optimiser) override;

	void ZeroGradients() override;

	WeightBias& GetParameters() {
		return parameters;
	}
};

#endif
