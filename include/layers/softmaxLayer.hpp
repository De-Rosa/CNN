#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include <layers/layer.hpp>
#include <cmath>

inline Matrix Softmax(Matrix& mat) {
	Matrix numerator = mat.unaryExpr(&std::exp)
	double sum = numerator.sum();
	return numerator.array() / sum;
}

class ReLULayer : public Layer {
public:
	Matrix FeedForward(Matrix& mat) override {
		return Softmax(mat);
	};

	Matrix FeedBackward(Matrix& mat, Matrix& grad) override {
		// where grad is prediction - true;
		return grad;
	};
};

#endif
