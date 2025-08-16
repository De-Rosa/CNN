#ifndef ADAMLAYER_H 
#define ADAMLAYER_H

#include <layers/layer.hpp>

struct WeightBias {
	Matrix weights;
	Matrix biases;

	WeightBias();
	WeightBias(int inputDim, int outputDim, double initValue);
};

struct AdamVariables {
	WeightBias grads; // dl/dw, dl/db
	WeightBias m; // first moment
	WeightBias v; // second moment
	
	AdamVariables(int inputDim, int outputDim);
};

class AdamLayer : public Layer {
protected:
	AdamVariables adamVars;
public: 
	AdamLayer(int inputDim, int outputDim);

	AdamVariables& GetAdamVars() {
		return adamVars;
	};
};

#endif
