#ifndef RELULAYER_H
#define RELULAYER_H

#include "layers/layer.hpp"

class ReLULayer : public Layer {
public:
	Matrix FeedForward(Matrix& mat) override;
	Matrix FeedBackward(Matrix& mat, Matrix& grad) override;
};

#endif
