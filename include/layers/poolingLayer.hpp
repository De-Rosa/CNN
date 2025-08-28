#ifndef POOLLAYER_H
#define POOLLAYER_H

#include "layers/layer.hpp"

class MaxPoolingLayer2D : public Layer {
public:
	MaxPoolingLayer2D(int size, int stride);

	Matrix FeedForward(const Matrix& mat) const;
	Matrix FeedBackward(const Matrix& mat, const Matrix& grad);

private:
	const int size, stride;
};

class MaxPoolingLayer3D : public Layer3D {
public:
	MaxPoolingLayer3D(int size, int stride);

	Matrix3D FeedForward(const Matrix3D& mat) const;
	Matrix3D FeedBackward(const Matrix3D& mat, const Matrix3D& grad);

private:
	const int size, stride;
};

#endif
