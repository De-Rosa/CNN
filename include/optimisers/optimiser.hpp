#ifndef OPTIMISERS_H 
#define OPTIMISERS_H

#include <algorithm>
#include <cmath>

class DenseLayer;

class Optimiser {
public:
	virtual ~Optimiser() = default;
	virtual void Update(DenseLayer& layer) = 0;
};

#endif
