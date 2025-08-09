#ifndef SGD_H 
#define SGD_H

#include <algorithm>
#include <layers/denseLayer.hpp>
#include <optimisers/optimiser.hpp>

// Stochastic Gradient Descent
class SGDOptimiser : public Optimiser {
	double stepSize;
public:
	SGDOptimiser(double stepSize = 0.001)
	: stepSize(stepSize) {}

	void Update(DenseLayer& denseLayer) {
		WeightBias& parameters = denseLayer.GetParameters();
		AdamVariables& vars = denseLayer.GetAdamVars();
		
		parameters.weights -= stepSize * vars.grads.weights;
		parameters.biases -= stepSize * vars.grads.biases;
	};
};

#endif
