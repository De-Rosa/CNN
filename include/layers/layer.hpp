#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

#include <vector>

using Matrix = Eigen::MatrixXd;
using Matrix3D = std::vector<Matrix>;

class Optimiser;

class Layer {
public:
	virtual ~Layer() {};

	virtual Matrix FeedForward(const Matrix& mat) const = 0;
	virtual Matrix FeedBackward(const Matrix& mat, const Matrix& grad) = 0;

	virtual void ZeroGradients() {}

	virtual void Optimise(Optimiser& optimiser) {}
};

class Layer3D {
public:
	virtual ~Layer3D() {};

	virtual Matrix3D FeedForward(const Matrix3D& mat) const = 0;
	virtual Matrix3D FeedBackward(const Matrix3D& mat, const Matrix3D& grad) = 0;

	virtual void ZeroGradients() {}

	virtual void Optimise(Optimiser& optimiser) {}
};

#endif
