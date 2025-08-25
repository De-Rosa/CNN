#include "layers/reLULayer.hpp"
#include <algorithm>

inline double ReLU(double x) {
	return std::max(0.0, x);
}

inline double ReLUDerivative(double x) {
	return x > 0.0;
}

Matrix ReLULayer::FeedForward(const Matrix& mat) const {
    return mat.unaryExpr(&ReLU);
};

Matrix ReLULayer::FeedBackward(const Matrix& mat, const Matrix& grad) {
    Matrix mask = mat.unaryExpr(&ReLUDerivative);
    return (grad.array() * mask.array()).matrix();
};
