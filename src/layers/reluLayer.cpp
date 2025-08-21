#include "layers/reLULayer.hpp"
#include <algorithm>

inline double ReLU(double x) {
	return std::max(0.0, x);
}

inline double ReLUDerivative(double x) {
	return x > 0.0;
}

Matrix ReLULayer::FeedForward(Matrix& mat) {
    return mat.unaryExpr(&ReLU);
};

Matrix ReLULayer::FeedBackward(Matrix& mat, Matrix& grad) {
    Matrix mask = mat.unaryExpr(&ReLUDerivative);
    return (grad.array() * mask.array()).matrix();
};
