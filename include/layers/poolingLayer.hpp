#ifndef POOLLAYER_H
#define POOLLAYER_H

#include "layers/layer.hpp"

class MaxPoolingLayer : public Layer {
public:
	MaxPoolingLayer(int size, int stride);

	Matrix FeedForward(const Matrix& mat) const;
	Matrix FeedBackward(const Matrix& mat, const Matrix& grad);

private:
	const int size, stride;
};

#endif
