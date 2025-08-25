#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "layers/layer.hpp"

class ConvLayer : public Layer {
public:
	Matrix FeedForward(const Matrix& mat) const;
	Matrix FeedBackward(const Matrix& mat, const Matrix& grad);
};

#endif
