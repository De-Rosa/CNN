#include "layers/poolingLayer.hpp"
#include <stdexcept>

MaxPoolingLayer::MaxPoolingLayer(int size, int stride)
	: size(size)
	, stride(stride)
{}


// https://blog.mlreview.com/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
Matrix MaxPoolingLayer::FeedForward(const Matrix& mat) const {
	if (size > mat.rows() || size > mat.cols()) throw std::runtime_error("kernel size of pooling layer too large for input matrix");

	// no padding, will cut off edges (valid pooling)
	int outputHeight = ((mat.rows() - size) / stride) + 1;
	int outputWidth = ((mat.cols() - size) / stride) + 1;

	Matrix outputMatrix(outputHeight, outputWidth);

	for (int h = 0; h < outputHeight; ++h) {
		for (int w = 0; w < outputWidth; ++w) {
			outputMatrix(h, w) = mat.block(stride * h, stride * w, size, size).maxCoeff();
		}
	}

	return outputMatrix;
}

Matrix MaxPoolingLayer::FeedBackward(const Matrix& mat, const Matrix& grad) {
	if (size > mat.rows() || size > mat.cols()) throw std::runtime_error("kernel size of pooling layer too large for input matrix");

	int outputHeight = ((mat.rows() - size) / stride) + 1;
	int outputWidth = ((mat.cols() - size) / stride) + 1;

	Matrix outputMatrix = Matrix::Zero(mat.rows(), mat.cols());

	for (int h = 0; h < outputHeight; ++h) {
		for (int w = 0; w < outputWidth; ++w) {
			Eigen::Index row, col;
			// will pick first occurrence if duplicate of max coeff
			double max = mat.block(stride * h, stride * w, size, size).maxCoeff(&row, &col);

			int newRow = h * stride + row;
			int newCol = w * stride + col;
			outputMatrix(newRow, newCol) = grad(h, w);
		}
	}

	return outputMatrix;
}
