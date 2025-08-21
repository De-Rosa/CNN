#ifndef SGD_H 
#define SGD_H

#include "layers/denseLayer.hpp"
#include "optimisers/optimiser.hpp"

// Stochastic Gradient Descent
class SGDOptimiser : public Optimiser {
	double stepSize;
public:
	SGDOptimiser(double stepSize = 0.001);

	void Update(DenseLayer& denseLayer);
};

#endif
