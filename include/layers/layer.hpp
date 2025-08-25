#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

using Matrix = Eigen::MatrixXd;

class Optimiser;

class Layer {
public:
	virtual ~Layer() {};

	virtual Matrix FeedForward(const Matrix& mat) const = 0;
	virtual Matrix FeedBackward(const Matrix& mat, const Matrix& grad) = 0;

	virtual void ZeroGradients() {}

	virtual void Optimise(Optimiser& optimiser) {}
};

#endif
