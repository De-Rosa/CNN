#ifndef ADAMLAYER_H 
#define ADAMLAYER_H

#include <layers/layer.hpp>

struct WeightBias {
	Matrix weights;
	Matrix biases;

	WeightBias() = default;

	WeightBias(int inputDim, int outputDim, double initValue) {
		if (inputDim <= 0 || outputDim <= 0) throw std::runtime_error("invalid dimensions");
		weights = Matrix::Constant(inputDim, outputDim, initValue);
		biases = Matrix::Constant(1, outputDim, initValue);
	}
};

struct AdamVariables {
	WeightBias grads; // dl/dw, dl/db
	WeightBias m; // first moment
	WeightBias v; // second moment
	
	AdamVariables(int inputDim, int outputDim)
	: grads(inputDim, outputDim, 0),
	  m(inputDim, outputDim, 0),
	  v(inputDim, outputDim, 0) {}
};

class AdamLayer : public Layer {
protected:
	AdamVariables adamVars;
public: 
	AdamLayer(int inputDim, int outputDim)
	: adamVars(inputDim, outputDim) {}

	AdamVariables& GetAdamVars() {
		return adamVars;
	};
};

#endif
