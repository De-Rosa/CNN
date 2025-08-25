#ifndef NETWORK_H 
#define NETWORK_H

#include "layers/layer.hpp"
#include "hyperparameters.hpp"
#include "optimisers/optimiser.hpp"
#include <memory>

using Layers = std::vector<std::unique_ptr<Layer>>;

class Network {
	const Layers layers;
	std::vector<Matrix> cache;

public:
	Network(Layers&& layers);

	Matrix FeedForward(const Matrix& mat);
	Matrix FeedBackward(const Matrix& grad);

	void Optimise(Optimiser& optimiser);

	void ZeroGradients();
};

#endif
