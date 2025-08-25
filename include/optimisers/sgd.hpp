#ifndef SGD_H 
#define SGD_H

#include "optimisers/optimiser.hpp"
#include "layers/denseLayer.hpp"

// Stochastic Gradient Descent
class SGDOptimiser : public Optimiser {
public:
	SGDOptimiser(double stepSize = 0.001);

	void Update(DenseLayer& denseLayer);

private:
	const double stepSize;
};

#endif
