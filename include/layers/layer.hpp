#ifndef LAYER_H
#def LAYER_H

class Layer {
	public:
	virtual ~Layer() {};
	virtual void FeedForward() = 0;
	virtual void FeedBackward() = 0;
};

#endif
