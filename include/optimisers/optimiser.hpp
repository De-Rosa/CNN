#ifndef OPTIMISERS_H 
#define OPTIMISERS_H

class DenseLayer;

class Optimiser {
public:
	virtual ~Optimiser() = default;
	virtual void Update(DenseLayer& layer) = 0;
};

#endif
