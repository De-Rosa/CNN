#include "layers/poolingLayer.hpp"

#include <stdexcept>

Matrix MaxPoolingForward(const Matrix& mat, int size, int stride) {
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

Matrix MaxPoolingBackward(const Matrix& mat, const Matrix& grad, int size, int stride) {
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

MaxPoolingLayer2D::MaxPoolingLayer2D(int size, int stride)
	: size(size)
	, stride(stride)
{}

// https://blog.mlreview.com/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
Matrix MaxPoolingLayer2D::FeedForward(const Matrix& mat) const {
	if (size > mat.rows() || size > mat.cols()) throw std::runtime_error("kernel size of pooling layer too large for input matrix");

	return MaxPoolingForward(mat, size, stride);
}

Matrix MaxPoolingLayer2D::FeedBackward(const Matrix& mat, const Matrix& grad) {
	if (size > mat.rows() || size > mat.cols()) throw std::runtime_error("kernel size of pooling layer too large for input matrix");

	return MaxPoolingBackward(mat, grad, size, stride);
}

MaxPoolingLayer3D::MaxPoolingLayer3D(int size, int stride)
	: size(size)
	, stride(stride)
{}

Matrix3D MaxPoolingLayer3D::FeedForward(const Matrix3D& mat) const {
	Matrix3D outputMatrix;
	outputMatrix.reserve(mat.size());

	for (const auto& channel : mat) {
		if (size > channel.rows() || size > channel.cols()) throw std::runtime_error("kernel size of pooling layer too large for a channel");
		
		// https://groups.google.com/a/chromium.org/g/chromium-dev/c/7mJypsYz6AA?pli=1
		outputMatrix.push_back(MaxPoolingForward(channel, size, stride));
	}

	return outputMatrix;
}

Matrix3D MaxPoolingLayer3D::FeedBackward(const Matrix3D& mat, const Matrix3D& grad) {
	if (mat.size() != grad.size()) throw std::runtime_error("input matrix size not same as grad matrix size");

	Matrix3D outputMatrix;
	outputMatrix.reserve(mat.size());

	for (size_t c = 0; c < mat.size(); ++c) {
		const auto& channel = mat[c];
		const auto& channelGrad = grad[c];

		if (size > channel.rows() || size > channel.cols()) throw std::runtime_error("kernel size of pooling layer too large for a channel");

		outputMatrix.push_back(MaxPoolingBackward(channel, channelGrad, size, stride));
	}
	
	return outputMatrix;
}
