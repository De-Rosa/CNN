#ifndef CONVLAYER_H
#define CONVLAYER_H

typedef Eigen::MatrixXd Matrix;

class ConvLayer : public Layer {
	Matrix FeedForward(Matrix mat) {};
	Matrix FeedBackward(Matrix mat) {};
};

#endif
