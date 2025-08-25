#include "layers/softmaxLayer.hpp"
#include <cmath>

using Array = Eigen::ArrayXXd;

inline Matrix Softmax(const Matrix& mat) {
    Array exp = mat.array().exp();
    return exp / exp.sum();
}

Matrix SoftmaxLayer::FeedForward(const Matrix& mat) const {
    return Softmax(mat);
};

Matrix SoftmaxLayer::FeedBackward(const Matrix& mat, const Matrix& grad) {
    // where grad is prediction - true;
    return grad;
};
