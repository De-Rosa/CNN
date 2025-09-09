#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "layers/layer.hpp"

class ConvLayer : public Layer3D {
public:
	ConvLayer(int channelCount, int filterCount, int filterSize, int stride);

	Matrix3D FeedForward(const Matrix3D& mat) const;
	Matrix3D FeedBackward(const Matrix3D& mat, const Matrix3D& grad);

private:
	const int channelCount, stride;
    const std::vector<Matrix3D> filters;
};

#endif
