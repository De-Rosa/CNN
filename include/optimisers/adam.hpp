#ifndef ADAM_H 
#define ADAM_H

#include <algorithm>
#include <cmath>
#include <layers/denseLayer.hpp>
#include <optimisers/optimiser.hpp>

class AdamOptimiser : public Optimiser {
	double decay1, decay2, stepSize, epsilon;
	int time = 0;
public:
	AdamOptimiser(double decay1 = 0.9, double decay2 = 0.999, double stepSize = 0.001, double epsilon = 1e-8)
	: decay1(decay1), decay2(decay2), stepSize(stepSize), epsilon(epsilon) {}

	// https://optimization.cbe.cornell.edu/index.php?title=Adam
	void Update(DenseLayer& denseLayer) {
		WeightBias& parameters = denseLayer.GetParameters();
		AdamVariables& vars = denseLayer.GetAdamVars();

		time++;
		
		// momentum
		vars.m.weights = decay1 * vars.m.weights +
			(1.0 - decay1) * vars.grads.weights;
		vars.m.biases  = decay1 * vars.m.biases +
			(1.0 - decay1) * vars.grads.biases;

		// rms
		vars.v.weights = decay2 * vars.v.weights +
			(1.0 - decay2) * vars.grads.weights.array().square().matrix();
		vars.v.biases  = decay2 * vars.v.biases +
			(1.0 - decay2) * vars.grads.biases.array().square().matrix();

		// bias correction
		Matrix corr_mw = vars.m.weights / (1.0 - std::pow(decay1, time));
		Matrix corr_mb = vars.m.biases  / (1.0 - std::pow(decay1, time));

		Matrix corr_vw = vars.v.weights / (1.0 - std::pow(decay2, time));
		Matrix corr_vb = vars.v.biases  / (1.0 - std::pow(decay2, time));

		// Update parameters
		parameters.weights -= stepSize *
			(corr_mw.array() / (corr_vw.array().sqrt() + epsilon)).matrix();
		parameters.biases  -= stepSize *
			(corr_mb.array() / (corr_vb.array().sqrt() + epsilon)).matrix();
	};
};

#endif
