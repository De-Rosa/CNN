#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

typedef Eigen::MatrixXd Matrix;

class Layer {
public:
	virtual ~Layer() {};
	virtual Matrix FeedForward(Matrix mat) = 0;
	virtual Matrix FeedBackward(Matrix mat) = 0;
};

#endif
