#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

using Matrix = Eigen::MatrixXd;

class Optimiser;

class Layer {
public:
	virtual ~Layer() {};

	virtual Matrix FeedForward(Matrix& mat) = 0;
	virtual Matrix FeedBackward(Matrix& mat, Matrix& grad) = 0;

	virtual void ZeroGradients() {}

	virtual void Optimise(Optimiser& optimiser) {}
};

#endif
