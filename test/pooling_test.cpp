#include <Eigen/Dense>
#include <layers/poolingLayer.hpp>
#include <gtest/gtest.h>

Matrix generatePoolTestMatrix() {
	Matrix mat(4,4);
	mat << 7, 3, 5, 2, 8, 7, 1, 6, 4, 9, 3, 9, 0, 8, 4, 5;
	return mat;
}

Matrix3D generate3DPoolTestMatrix() {
	Matrix3D mat = { generatePoolTestMatrix(), generatePoolTestMatrix() };
	return mat;
}

TEST(MaxPoolForwardPass, PositiveValues) {
	Matrix mat = generatePoolTestMatrix();

	MaxPoolingLayer2D layer(2, 2);
	Matrix result = layer.FeedForward(mat);

	Matrix expected(2, 2);
	expected << 8, 6, 9, 9;

	ASSERT_TRUE(result.isApprox(expected));
};

TEST(MaxPoolBackwardPass, PositiveValues) {
	Matrix mat = generatePoolTestMatrix();
	Matrix grad(2, 2);
	grad << 1, 2, 3, 4;

	MaxPoolingLayer2D layer(2, 2);
	Matrix result = layer.FeedBackward(mat, grad);

	Matrix expected(4, 4);
	expected << 0, 0, 0, 0, 1, 0, 0, 2, 0, 3, 0, 4, 0, 0, 0, 0;

	ASSERT_TRUE(result.isApprox(expected));
};

TEST(MaxPool3DForwardPass, PositiveValues) {
	Matrix3D mat = generate3DPoolTestMatrix();

	MaxPoolingLayer3D layer(2, 2);
	Matrix3D result = layer.FeedForward(mat);

	Matrix expected(2, 2);
	expected << 8, 6, 9, 9;

	ASSERT_EQ(result.size(), mat.size());

	for (size_t c = 0; c < result.size(); ++c) {
		ASSERT_TRUE(result[c].isApprox(expected));
	}

};

TEST(MaxPool3DBackwardPass, PositiveValues) {
	Matrix3D mat = generate3DPoolTestMatrix();

	// https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialAdvancedInitialization.html
	Matrix3D grad = { (Matrix(2,2) << 1, 2, 3, 4).finished(),
			(Matrix(2,2) << 1, 2, 3, 4).finished() };

	MaxPoolingLayer3D layer(2, 2);
	Matrix3D result = layer.FeedBackward(mat, grad);

	Matrix expected(4, 4);
	expected << 0, 0, 0, 0, 1, 0, 0, 2, 0, 3, 0, 4, 0, 0, 0, 0;

	ASSERT_EQ(result.size(), mat.size());

	for (size_t c = 0; c < result.size(); ++c) {
		ASSERT_TRUE(result[c].isApprox(expected));
	}
}
