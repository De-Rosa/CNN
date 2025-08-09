#ifndef NETWORK_H 
#define NETWORK_H

#include <layers/layer.hpp>
#include <hyperparameters.hpp>
#include <optimisers/optimiser.hpp>
#include <memory>

using Layers = std::vector<std::unique_ptr<Layer>>;

class Network {
	Layers layers;
	std::vector<Matrix> cache;

public:
	Network(Layers&& layers) 
	: layers(std::move(layers)) {}

	Matrix FeedForward(Matrix& mat) { 
		// cached layer at index i is input to layer i during forward propagation
		cache.clear();
		cache.push_back(mat);

		Matrix current = mat;
		for (auto& layer : layers) {
			current = layer->FeedForward(current);
			cache.push_back(current);
		}

		return current;
	};

	Matrix FeedBackward(Matrix& grad) {
		Matrix current = grad;
		for (int i = layers.size() - 1; i >= 0; --i) {
			current = layers[i]->FeedBackward(cache[i], current);
		}
		return current;
	};

	void Optimise(Optimiser& optimiser) {
		for (auto& layer : layers) {
			layer->Optimise(optimiser);
		}
	}

	void ZeroGradients() {
		for (auto& layer : layers) {
			layer->ZeroGradients();
		}
	}
};

#endif
