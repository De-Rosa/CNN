#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <layers/layer.hpp>

class ConvLayer : public Layer {
	Matrix FeedForward(Matrix mat);
	Matrix FeedBackward(Matrix mat);
};

#endif
