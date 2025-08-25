#ifndef RELULAYER_H
#define RELULAYER_H

#include "layers/layer.hpp"

class ReLULayer : public Layer {
public:
	Matrix FeedForward(const Matrix& mat) const override;
	Matrix FeedBackward(const Matrix& mat, const Matrix& grad) override;
};

#endif
