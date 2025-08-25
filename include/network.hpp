#ifndef NETWORK_H 
#define NETWORK_H

#include <memory>

#include "hyperparameters.hpp"
#include "layers/layer.hpp"
#include "optimisers/optimiser.hpp"

using Layers = std::vector<std::unique_ptr<Layer>>;

class Network {
public:
	Network(Layers&& layers);

	Matrix FeedForward(const Matrix& mat);
	Matrix FeedBackward(const Matrix& grad);

	void Optimise(Optimiser& optimiser);

	void ZeroGradients();

private:
	const Layers layers;
	std::vector<Matrix> cache;
};

#endif
