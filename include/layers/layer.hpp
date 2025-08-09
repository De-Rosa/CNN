#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

typedef Eigen::MatrixXd Matrix;

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
