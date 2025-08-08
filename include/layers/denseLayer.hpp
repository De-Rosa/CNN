#ifndef DENSELAYER_H 
#define DENSELAYER_H

#include <layers/layer.hpp>
#include <hyperparameters.hpp>
#include <algorithm>

typedef Eigen::MatrixXd Matrix;

struct WeightBias {
	Matrix weights;
	Matrix biases;
};

struct AdamVariables {
	WeightBias grads; // dl/dw, dl/db
	WeightBias m; // first moment
	WeightBias v; // second moment
};

class DenseLayer : public Layer {
	WeightBias parameters;
	AdamVariables adamVars;

public:
	Matrix FeedForward(Matrix& mat) {
	};

	Matrix FeedBackward(Matrix& mat) {
	};

	void Adam();
};

#endif
