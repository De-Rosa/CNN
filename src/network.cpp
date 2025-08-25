#include "network.hpp"

#include <memory>

Network::Network(Layers&& layers)
        : layers(std::move(layers))
{}

Matrix Network::FeedForward(const Matrix& mat) { 
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

Matrix Network::FeedBackward(const Matrix& grad) {
    Matrix current = grad;
    for (int i = layers.size() - 1; i >= 0; --i) {
        current = layers[i]->FeedBackward(cache[i], current);
    }
    return current;
};

void Network::Optimise(Optimiser& optimiser) {
    for (auto& layer : layers) {
        layer->Optimise(optimiser);
    }
}

void Network::ZeroGradients() {
    for (auto& layer : layers) {
        layer->ZeroGradients();
    }
}
