#ifndef RELULAYER_H
#define RELULAYER_H

#include <layers/layer.hpp>
#include <algorithm>

inline double ReLU(double x) {
	return std::max(0.0, x);
}

inline double ReLUDerivative(double x) {
	return x > 0.0;
}

class ReLULayer : public Layer {
public:
	Matrix FeedForward(Matrix& mat) override {
		return mat.unaryExpr(&ReLU);
	};

	Matrix FeedBackward(Matrix& mat, Matrix& grad) override {
		Matrix mask = mat.unaryExpr(&ReLUDerivative);
		return (grad.array() * mask.array()).matrix();
	};
};

#endif
