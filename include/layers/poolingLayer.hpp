#ifndef POOLLAYER_H
#define POOLLAYER_H

#include "layers/layer.hpp"

class MaxPoolingLayer : public Layer {
private:
	int size, stride;
public:
	MaxPoolingLayer(int size, int stride);

	Matrix FeedForward(Matrix& mat);
	Matrix FeedBackward(Matrix& mat, Matrix& grad);
};

#endif
