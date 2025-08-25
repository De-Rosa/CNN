#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include "layers/layer.hpp"

class SoftmaxLayer : public Layer {
public:
	Matrix FeedForward(const Matrix& mat) const override;
	Matrix FeedBackward(const Matrix& mat, const Matrix& grad) override;
};

#endif
