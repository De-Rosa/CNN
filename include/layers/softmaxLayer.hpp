#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include <layers/layer.hpp>

class SoftmaxLayer : public Layer {
public:
	Matrix FeedForward(Matrix& mat) override;
	Matrix FeedBackward(Matrix& mat, Matrix& grad) override;
};

#endif
