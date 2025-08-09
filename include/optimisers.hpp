#ifndef OPTIMISERS_H 
#define OPTIMISERS_H

#include <layers/denseLayer.hpp>
#include <hyperparameters.hpp>
#include <algorithm>
#include <cmath>

class Adam {
	double decay1, decay2, stepSize, epsilon;
	int time = 0;
public:
	Adam(double decay1 = 0.9, double decay2 = 0.999, double stepSize = 0.001, double epsilon = 1e-8)
	: decay1(decay1), decay2(decay2), stepSize(stepSize), epsilon(epsilon) {}

	// https://optimization.cbe.cornell.edu/index.php?title=Adam
	void Update(WeightBias& parameters, AdamVariables& vars) {
		time++;
		
		// momentum
		vars.m.weights = hyperparameters.decay1 * vars.m.weights +
			(1.0 - hyperparameters.decay1) * vars.grads.weights;
		vars.m.biases  = hyperparameters.decay1 * vars.m.biases +
			(1.0 - hyperparameters.decay1) * vars.grads.biases;

		// rms
		vars.v.weights = hyperparameters.decay2 * vars.v.weights +
			(1.0 - hyperparameters.decay2) * vars.grads.weights.array().square().matrix();
		vars.v.biases  = hyperparameters.decay2 * vars.v.biases +
			(1.0 - hyperparameters.decay2) * vars.grads.biases.array().square().matrix();

		// bias correction
		Matrix corr_mw = vars.m.weights / (1.0 - std::pow(hyperparameters.decay1, time));
		Matrix corr_mb = vars.m.biases  / (1.0 - std::pow(hyperparameters.decay1, time));

		Matrix corr_vw = vars.v.weights / (1.0 - std::pow(hyperparameters.decay2, time));
		Matrix corr_vb = vars.v.biases  / (1.0 - std::pow(hyperparameters.decay2, time));

		// Update parameters
		parameters.weights -= hyperparameters.stepSize *
				  (corr_mw.array() / (corr_vw.array().sqrt() + hyperparameters.epsilon)).matrix();
		parameters.biases  -= hyperparameters.stepSize *
				  (corr_mb.array() / (corr_vw.array().sqrt() + hyperparameters.epsilon)).matrix();
	};
};

#endif
