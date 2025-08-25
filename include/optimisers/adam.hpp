#ifndef ADAM_H 
#define ADAM_H

#include <unordered_map>

#include "optimisers/optimiser.hpp"
#include "layers/denseLayer.hpp"

struct AdamState {
	WeightBias m; // first moment
	WeightBias v; // second moment
	
	AdamState(int inputDim, int outputDim);
};

class AdamOptimiser : public Optimiser {
public:
	AdamOptimiser(double decay1 = 0.9, double decay2 = 0.999, double stepSize = 0.001, double epsilon = 1e-8);
	
	void Update(DenseLayer& denseLayer);

private:
	AdamState& GetState(DenseLayer& denseLayer);

	std::unordered_map<DenseLayer*, AdamState> states;
	const double decay1, decay2, stepSize, epsilon;
	int time = 0;
};

#endif
