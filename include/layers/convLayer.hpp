#ifndef CONVLAYER_H
#define CONVLAYER_H

class ConvLayer : public Layer {
	Matrix FeedForward(Matrix mat) {};
	Matrix FeedBackward(Matrix mat) {};
};

#endif
