#include "layers/softmaxLayer.hpp"
#include <cmath>

using Array = Eigen::ArrayXXd;

inline Matrix Softmax(Matrix& mat) {
    Array exp = mat.array().exp();
    return exp / exp.sum();
}

Matrix SoftmaxLayer::FeedForward(Matrix& mat) {
    return Softmax(mat);
};

Matrix SoftmaxLayer::FeedBackward(Matrix& mat, Matrix& grad) {
    // where grad is prediction - true;
    return grad;
};
