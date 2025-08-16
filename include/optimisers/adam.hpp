#ifndef ADAM_H 
#define ADAM_H

#include <layers/denseLayer.hpp>
#include <optimisers/optimiser.hpp>

class AdamOptimiser : public Optimiser {
	double decay1, decay2, stepSize, epsilon;
	int time = 0;
public:
	AdamOptimiser(double decay1 = 0.9, double decay2 = 0.999, double stepSize = 0.001, double epsilon = 1e-8);

	void Update(DenseLayer& denseLayer);
};

#endif
